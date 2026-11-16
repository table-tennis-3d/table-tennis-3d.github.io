// 推定結果（1コマごとの33点）を、3D表示と分析に使える「動き」データに整える
import { N, LM, MIRROR_INDEX } from './landmarks.js';

// なめらかさ設定 → 平滑化の幅（秒）。止まっているときは max、速く動いているときは min に近づく
const SMOOTH_LEVELS = [
  null,
  { max: 0.035, min: 0.008 },
  { max: 0.060, min: 0.008 },
  { max: 0.100, min: 0.012 },
];
// この速さ（m/秒）を超える動きは、ほとんど平滑化しない（スイングの山を削らないため）
const ACTIVE_SPEED = 0.8;
// 奥行き（カメラからの前後）は1台のカメラでは推定が不安定で、1コマで大きく飛ぶことがある。
// 奥行きだけは平滑化の幅を広げて、飛びが速さの山として出にくいようにする
const DEPTH_WIDEN = 1.6;
const DEPTH_MIN_SECONDS = 0.045;
// かかと・つま先を床からどれだけ浮かせるか（人形の足の厚み分）
const FOOT_CLEARANCE = 0.012;
const FOOT_POINTS = [LM.L_HEEL, LM.R_HEEL, LM.L_FOOT, LM.R_FOOT];
const S = N * 3; // 1コマあたりの3D座標の数

// 体の横移動の推定に使う点（体幹と手足の付け根〜足首）
const ROOT_POINTS = [LM.L_SHOULDER, LM.R_SHOULDER, LM.L_HIP, LM.R_HIP, LM.L_KNEE, LM.R_KNEE, LM.L_ANKLE, LM.R_ANKLE];

/**
 * @param {object} raw 推定結果 { fps, start, frames: [{t, world, image, vis}], videoWidth, videoHeight }
 * @param {{mirror?:boolean, smoothing?:number, floor?:boolean}} opts
 */
export function buildMotion(raw, opts = {}) {
  const fps = raw.fps;
  const frames = raw.frames;
  const n = frames.length;
  const mirror = !!opts.mirror;

  const pos = new Float32Array(n * S);
  const img = new Float32Array(n * N * 2);
  const vis = new Float32Array(n * N);
  const valid = new Uint8Array(n);
  const times = new Float64Array(n);
  let validCount = 0;

  for (let i = 0; i < n; i++) {
    const f = frames[i];
    times[i] = f.t;
    if (!f.world) continue;
    valid[i] = 1;
    validCount++;
    for (let j = 0; j < N; j++) {
      const src = mirror ? MIRROR_INDEX[j] : j;
      const x = f.world[src * 3], y = f.world[src * 3 + 1], z = f.world[src * 3 + 2];
      // MediaPipe（x右・y下・z奥）→ three.js（x右・y上・z手前）
      pos[i * S + j * 3] = mirror ? -x : x;
      pos[i * S + j * 3 + 1] = -y;
      pos[i * S + j * 3 + 2] = -z;
      // 画像上の位置は元の向きのまま（表示側で左右反転する）
      img[(i * N + j) * 2] = f.image[src * 3];
      img[(i * N + j) * 2 + 1] = f.image[src * 3 + 1];
      vis[i * N + j] = f.vis[src];
    }
  }

  if (validCount === 0) {
    const err = new Error('人物を検出できませんでした');
    err.code = 'NO_PERSON';
    throw err;
  }

  fillGaps(pos, S, valid, n);
  fillGaps(img, N * 2, valid, n);
  fillGaps(vis, N, valid, n);

  // 平滑化（位置）と速度
  const level = SMOOTH_LEVELS[opts.smoothing ?? 2];
  let P, V;
  let I = img;
  if (level) {
    const bw = bandwidths(pos, n, fps, level);
    ({ pos: P, vel: V } = localQuadratic(pos, n, 3, bw, vis, valid, fps));
    I = localQuadratic(img, n, 2, bw, vis, valid, fps).pos;
  } else {
    P = pos;
    V = centralDifference(pos, n, fps);
  }

  // 床の高さをそろえる（腰が原点のデータなので、しゃがむと足が上がってしまうのを直す）
  const footMin = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let m = Infinity;
    for (const j of FOOT_POINTS) m = Math.min(m, P[i * S + j * 3 + 1]);
    footMin[i] = m;
  }
  let offsets;
  if (opts.floor !== false) {
    const smoothed = gaussian1D(footMin, n, 0.12 * fps);
    offsets = smoothed.map((m) => FOOT_CLEARANCE - m);
  } else {
    const med = median(Array.from(footMin));
    offsets = new Float32Array(n).fill(FOOT_CLEARANCE - med);
  }
  for (let i = 0; i < n; i++) {
    const o = offsets[i];
    for (let j = 0; j < N; j++) P[i * S + j * 3 + 1] += o;
  }

  const facing = averageFacing(P, n, valid);
  const rootX = estimateRootX(raw, n, fps, valid, mirror);

  return {
    n,
    fps,
    start: times[0],
    end: times[n - 1],
    times,
    pos: P,
    vel: V,
    img: I,
    vis,
    valid,
    rootX,
    validRatio: validCount / n,
    facingYaw: facing.yaw,
    hipHeight: facing.hipHeight,
    videoWidth: raw.videoWidth || 0,
    videoHeight: raw.videoHeight || 0,
  };
}

/**
 * 時刻tの姿勢を補間して out(Float32Array(99)) に書き込む。範囲外は端のコマ。
 * withRoot: 体の横移動（体重移動・フットワーク）を足すかどうか
 */
export function samplePose(motion, t, out, withRoot = false) {
  const { n, fps, start, pos } = motion;
  let f = (t - start) * fps;
  if (!(f > 0)) f = 0;
  if (f > n - 1) f = n - 1;
  const i0 = Math.floor(f);
  const i1 = Math.min(n - 1, i0 + 1);
  const a = f - i0;
  const o0 = i0 * S, o1 = i1 * S;
  for (let k = 0; k < S; k++) out[k] = pos[o0 + k] * (1 - a) + pos[o1 + k] * a;
  const root = withRoot ? motion.rootX : null;
  if (root) {
    const rx = root[i0] * (1 - a) + root[i1] * a;
    for (let j = 0; j < N; j++) out[j * 3] += rx;
  }
  return out;
}

export function frameIndexAt(motion, t) {
  const f = Math.round((t - motion.start) * motion.fps);
  return Math.max(0, Math.min(motion.n - 1, f));
}

/** 時刻tにいちばん近いコマの画像上の位置（33点 × (u, v)）。動画のコマと1対1で対応する */
export function imageAt(motion, t) {
  const i = frameIndexAt(motion, t);
  return motion.img.subarray(i * N * 2, (i + 1) * N * 2);
}

export function isValidAt(motion, t) {
  return motion.valid[frameIndexAt(motion, t)] === 1;
}

// ---------- 内部処理 ----------

// 検出できなかったコマを前後のコマから直線補間する
function fillGaps(arr, stride, valid, n) {
  let prev = -1;
  for (let i = 0; i < n; i++) {
    if (!valid[i]) continue;
    if (prev === -1 && i > 0) {
      for (let k = 0; k < i; k++) arr.copyWithin(k * stride, i * stride, i * stride + stride);
    } else if (prev >= 0 && i - prev > 1) {
      for (let k = prev + 1; k < i; k++) {
        const a = (k - prev) / (i - prev);
        for (let c = 0; c < stride; c++) {
          arr[k * stride + c] = arr[prev * stride + c] * (1 - a) + arr[i * stride + c] * a;
        }
      }
    }
    prev = i;
  }
  if (prev >= 0) {
    for (let k = prev + 1; k < n; k++) arr.copyWithin(k * stride, prev * stride, prev * stride + stride);
  }
}

/**
 * 関節ごと・コマごとの平滑化の幅（コマ数）を決める。
 * 止まっている関節は広く平均してブレを消し、速く動いている関節はほとんど平均しない。
 */
export function bandwidths(pos, n, fps, level, activeSpeed = ACTIVE_SPEED) {
  const bw = new Float32Array(n * N);
  // 速さは約1/30秒の間隔で測る（コマ数が多い動画でもノイズで速く見えないように）
  const h = Math.max(1, Math.round(fps / 30));
  const sq = new Float32Array(n);
  const spread = 0.05 * fps; // 前後0.05秒くらいの動きを見て決める（切り返しの瞬間も速い動きとして扱う）
  const minFrames = Math.max(0.6, level.min * fps);
  const maxFrames = Math.max(minFrames, level.max * fps);
  for (let j = 0; j < N; j++) {
    for (let i = 0; i < n; i++) {
      const a = Math.max(0, i - h), b = Math.min(n - 1, i + h);
      if (a === b) { sq[i] = 0; continue; }
      const oa = a * S + j * 3, ob = b * S + j * 3;
      // 画面に映る方向（左右・上下）の動きだけで判断する。奥行きの飛びを「速い動き」と見なさないため
      const dx = pos[ob] - pos[oa], dy = pos[ob + 1] - pos[oa + 1];
      const k = fps / (b - a);
      sq[i] = (dx * dx + dy * dy) * k * k;
    }
    const act = gaussian1D(sq, n, spread);
    for (let i = 0; i < n; i++) {
      const r2 = act[i] / (activeSpeed * activeSpeed);
      bw[i * N + j] = minFrames + (maxFrames - minFrames) / (1 + r2);
    }
  }
  return bw;
}

/**
 * 局所2次回帰（ガウス重み・見え具合で重み付け）。位置と速度（1秒あたり）を同時に求める。
 * 単純な平均と違って、加速している動きの山や谷を削らない。位相のずれもない。
 * comps が 3 のとき、3番目（奥行き）だけは幅を広げる。
 */
export function localQuadratic(arr, n, comps, bw, vis, valid, fps) {
  const stride = N * comps;
  const out = new Float32Array(arr.length);
  const vel = new Float32Array(arr.length);
  const depthMin = DEPTH_MIN_SECONDS * fps;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < N; j++) {
      const sigma = bw[i * N + j];
      const o = i * stride + j * comps;
      fitAt(arr, n, stride, i, j * comps, comps === 3 ? 2 : comps, sigma, vis, valid, j, out, vel, o, fps);
      if (comps === 3) {
        fitAt(arr, n, stride, i, j * comps + 2, 1, Math.max(sigma * DEPTH_WIDEN, depthMin), vis, valid, j, out, vel, o + 2, fps);
      }
    }
  }
  return { pos: out, vel };
}

const _T0 = [0, 0], _T1 = [0, 0], _T2 = [0, 0];
// コマ i のまわりで、count 個の成分（arr の off 番目から）を2次式に当てはめる
function fitAt(arr, n, stride, i, off, count, sigma, vis, valid, j, out, vel, dst, fps) {
  const R = Math.max(2, Math.ceil(sigma * 3));
  const lo = Math.max(0, i - R), hi = Math.min(n - 1, i + R);
  const inv = -0.5 / (sigma * sigma);
  let S0 = 0, S1 = 0, S2 = 0, S3 = 0, S4 = 0;
  _T0[0] = _T0[1] = _T1[0] = _T1[1] = _T2[0] = _T2[1] = 0;
  for (let ii = lo; ii <= hi; ii++) {
    const k = ii - i;
    const w = Math.exp(inv * k * k) * (valid[ii] ? Math.max(vis[ii * N + j], 0.15) : 0.02);
    const wk = w * k, wk2 = wk * k;
    S0 += w; S1 += wk; S2 += wk2; S3 += wk2 * k; S4 += wk2 * k * k;
    const o = ii * stride + off;
    for (let c = 0; c < count; c++) {
      const y = arr[o + c];
      _T0[c] += w * y; _T1[c] += wk * y; _T2[c] += wk2 * y;
    }
  }
  // 端のコマ（片側にしかデータがない）は2次だと不安定なので1次にする
  const twoSided = i - lo >= 2 && hi - i >= 2;
  const A = S2 * S4 - S3 * S3, B = S1 * S4 - S2 * S3, C = S1 * S3 - S2 * S2;
  const det3 = S0 * A - S1 * B + S2 * C;
  const det2 = S0 * S2 - S1 * S1;
  for (let c = 0; c < count; c++) {
    let c0, c1;
    if (twoSided && det3 > 1e-9 * S0 * S2 * S4) {
      const D = _T1[c] * S4 - _T2[c] * S3;
      c0 = (_T0[c] * A - S1 * D + S2 * (_T1[c] * S3 - _T2[c] * S2)) / det3;
      c1 = (S0 * D - _T0[c] * B + S2 * (S1 * _T2[c] - S2 * _T1[c])) / det3;
    } else if (det2 > 1e-9 * S0 * S2) {
      c0 = (_T0[c] * S2 - S1 * _T1[c]) / det2;
      c1 = (S0 * _T1[c] - S1 * _T0[c]) / det2;
    } else {
      c0 = _T0[c] / S0;
      c1 = 0;
    }
    out[dst + c] = c0;
    vel[dst + c] = c1 * fps;
  }
}

// 平滑化なしのときの速度（前後のコマの差）
function centralDifference(pos, n, fps) {
  const vel = new Float32Array(pos.length);
  for (let i = 0; i < n; i++) {
    const a = Math.max(0, i - 1), b = Math.min(n - 1, i + 1);
    if (a === b) continue;
    const k = fps / (b - a);
    for (let c = 0; c < S; c++) vel[i * S + c] = (pos[b * S + c] - pos[a * S + c]) * k;
  }
  return vel;
}

export function gaussian1D(arr, n, sigmaFrames) {
  const out = new Float32Array(n);
  if (sigmaFrames <= 0.01) { out.set(arr.subarray ? arr.subarray(0, n) : arr.slice(0, n)); return out; }
  const R = Math.max(1, Math.ceil(sigmaFrames * 3));
  for (let i = 0; i < n; i++) {
    let ws = 0, acc = 0;
    for (let ii = Math.max(0, i - R); ii <= Math.min(n - 1, i + R); ii++) {
      const w = Math.exp(-0.5 * ((ii - i) / sigmaFrames) ** 2);
      ws += w;
      acc += w * arr[ii];
    }
    out[i] = acc / ws;
  }
  return out;
}

function median(values) {
  if (!values.length) return 0;
  const s = values.slice().sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

// 体の正面の向き（腰と肩のラインから）をクリップ全体で平均する
function averageFacing(P, n, valid) {
  let fx = 0, fz = 0, hy = 0, cnt = 0;
  for (let i = 0; i < n; i++) {
    if (!valid[i]) continue;
    const o = i * S;
    // 右→左の向きのベクトル
    const hx = P[o + LM.L_HIP * 3] - P[o + LM.R_HIP * 3];
    const hz = P[o + LM.L_HIP * 3 + 2] - P[o + LM.R_HIP * 3 + 2];
    const sx = P[o + LM.L_SHOULDER * 3] - P[o + LM.R_SHOULDER * 3];
    const sz = P[o + LM.L_SHOULDER * 3 + 2] - P[o + LM.R_SHOULDER * 3 + 2];
    // 正面 = (右→左) × 上 = (-z, 0, x)
    fx += -(hz * 2 + sz);
    fz += hx * 2 + sx;
    hy += (P[o + LM.L_HIP * 3 + 1] + P[o + LM.R_HIP * 3 + 1]) / 2;
    cnt++;
  }
  const yaw = (fx === 0 && fz === 0) ? 0 : Math.atan2(fx, fz);
  return { yaw, hipHeight: cnt ? hy / cnt : 0.95 };
}

/**
 * 体全体の横移動（カメラから見て左右）を推定する。体重移動やフットワークが見えるようになる。
 * 3Dの姿勢（腰が原点）を画像上の位置に当てはめて、1コマごとに「画面上の位置」と「縮尺」を求め、
 * 腰の中心が画面上で左右にどれだけ動いたかをメートルに直す。
 * 前後（奥行き）の移動は、映っている大きさの変化からしか分からず誤差が大きいので扱わない。
 * 返り値: Float32Array(n)（メートル・画面の右が＋・クリップ全体の中央値を0とする）
 */
export function estimateRootX(raw, n, fps, valid, mirror) {
  const W = raw.videoWidth, H = raw.videoHeight;
  if (!(W > 0 && H > 0)) return null;
  const X = new Float32Array(n);
  const ok = new Uint8Array(n);
  let count = 0;
  for (let i = 0; i < n; i++) {
    const f = raw.frames[i];
    if (!valid[i] || !f.world) continue;
    // 重み付きの最小二乗で、画像上の位置 ≒ 縮尺 × 3D位置(x, y) + 平行移動 を解く
    let sw = 0, mx = 0, my = 0, mu = 0, mv = 0;
    for (const j of ROOT_POINTS) {
      const w = Math.max(f.vis[j], 0.05);
      sw += w;
      mx += w * f.world[j * 3]; my += w * f.world[j * 3 + 1];
      mu += w * f.image[j * 3] * W; mv += w * f.image[j * 3 + 1] * H;
    }
    mx /= sw; my /= sw; mu /= sw; mv /= sw;
    let num = 0, den = 0;
    for (const j of ROOT_POINTS) {
      const w = Math.max(f.vis[j], 0.05);
      const dx = f.world[j * 3] - mx, dy = f.world[j * 3 + 1] - my;
      num += w * (dx * (f.image[j * 3] * W - mu) + dy * (f.image[j * 3 + 1] * H - mv));
      den += w * (dx * dx + dy * dy);
    }
    if (!(den > 1e-6) || !(num > 0)) continue;
    const scale = num / den; // 1メートルあたりの画素数
    const u0 = mu - scale * mx; // 腰の中心（3Dの原点）の画面上の位置
    X[i] = (u0 - W / 2) / scale;
    ok[i] = 1;
    count++;
  }
  if (count < 2) return null;
  fillGaps(X, 1, ok, n);
  const xs = gaussian1D(X, n, 0.06 * fps);
  const x0 = median(Array.from(xs));
  for (let i = 0; i < n; i++) xs[i] = mirror ? -(xs[i] - x0) : xs[i] - x0;
  return xs;
}
