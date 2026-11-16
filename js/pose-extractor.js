// 動画から1コマずつ姿勢（33点の3D座標）を推定する
import { MEDIAPIPE_WASM, WASM_SIMD_ONLY, MODEL_URLS, MODEL_FORMAT } from './config.js';
import { N } from './landmarks.js';

// Base64のテキストで置いたモデルを読み込んでバイト列に戻す（一度読んだら使い回す）
const modelBuffers = new Map();
function loadModelBuffer(model) {
  if (!modelBuffers.has(model)) {
    const p = fetch(MODEL_URLS[model])
      .then((r) => {
        if (!r.ok) throw new Error(`AIモデルを読み込めませんでした（${r.status}）`);
        return r.text();
      })
      .then(decodeBase64)
      .catch((err) => {
        modelBuffers.delete(model);
        throw err;
      });
    modelBuffers.set(model, p);
  }
  return modelBuffers.get(model);
}

function decodeBase64(text) {
  const clean = text.replace(/[^A-Za-z0-9+/=]/g, '');
  const pad = clean.endsWith('==') ? 2 : clean.endsWith('=') ? 1 : 0;
  const out = new Uint8Array((clean.length / 4) * 3 - pad);
  const CHUNK = 1 << 20; // 4の倍数ずつ区切ってメモリを節約
  let o = 0;
  for (let i = 0; i < clean.length; i += CHUNK) {
    const bin = atob(clean.slice(i, i + CHUNK));
    for (let k = 0; k < bin.length; k++) out[o++] = bin.charCodeAt(k);
  }
  return out;
}

let visionPromise = null;
function loadVision() {
  if (!visionPromise) {
    visionPromise = import('@mediapipe/tasks-vision').catch((err) => {
      visionPromise = null;
      throw err;
    });
  }
  return visionPromise;
}

// ソフトウェア描画（GPUが使えない環境）ではGPU処理がかえって遅いのでCPUを使う
function prefersCpu() {
  try {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2');
    if (!gl) return true;
    const ext = gl.getExtension('WEBGL_debug_renderer_info');
    const renderer = String(ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER));
    gl.getExtension('WEBGL_lose_context')?.loseContext();
    return /swiftshader|llvmpipe|softpipe|software|basic render/i.test(renderer);
  } catch {
    return true;
  }
}

function waitForEvent(target, name, timeoutMs) {
  return new Promise((resolve) => {
    let timer = null;
    const done = () => {
      target.removeEventListener(name, done);
      clearTimeout(timer);
      resolve();
    };
    target.addEventListener(name, done);
    timer = setTimeout(done, timeoutMs);
  });
}

async function seekTo(video, t) {
  const seeked = waitForEvent(video, 'seeked', 4000);
  video.currentTime = t;
  await seeked;
  if (video.readyState < 2) await waitForEvent(video, 'loadeddata', 2000);
}

// map: 範囲を切り抜いて推定したとき、動画全体の座標（0〜1）に戻すための係数
function toFrame(result, t, map) {
  const world = result?.worldLandmarks?.[0];
  const image = result?.landmarks?.[0];
  if (!world || !image || world.length < N) return { t, world: null, image: null, vis: null };
  const w = new Float32Array(N * 3);
  const im = new Float32Array(N * 3);
  const vis = new Float32Array(N);
  const ox = map ? map.ox : 0, oy = map ? map.oy : 0, kx = map ? map.kx : 1, ky = map ? map.ky : 1;
  for (let j = 0; j < N; j++) {
    w[j * 3] = world[j].x; w[j * 3 + 1] = world[j].y; w[j * 3 + 2] = world[j].z;
    im[j * 3] = ox + image[j].x * kx; im[j * 3 + 1] = oy + image[j].y * ky; im[j * 3 + 2] = image[j].z * kx;
    vis[j] = image[j].visibility ?? world[j].visibility ?? 1;
  }
  return { t, world: w, image: im, vis };
}

const COMMON_RATES = [10, 12, 12.5, 15, 20, 23.976, 24, 25, 29.97, 30, 48, 50, 59.94, 60, 90, 100, 119.88, 120, 239.76, 240];

/**
 * 動画の実際のコマ数（1秒あたり）を調べる。少しだけスロー再生して、表示されたコマの時刻を集める。
 * 分からないときは null（対応していないブラウザ、自動再生できない状況など）。
 * @returns {Promise<{fps:number, frame:number, phase:number}|null>} frame: 1コマの長さ（秒）、phase: あるコマの開始時刻
 */
export async function detectFrameRate(video, at = 0) {
  if (typeof video.requestVideoFrameCallback !== 'function') return null;
  const duration = Number.isFinite(video.duration) ? video.duration : 0;
  if (!(duration > 0.3)) return null;
  const from = Math.max(0, Math.min(at, duration - 0.6));
  const prevRate = video.playbackRate;
  const wasMuted = video.muted;
  const stamps = [];
  let handle = 0;
  let running = true;
  try {
    video.muted = true;
    await seekTo(video, from);
    video.playbackRate = 0.25; // ゆっくり再生して、コマ数の多い動画でも1コマずつ表示されるようにする
    const collect = (_now, meta) => {
      if (!running) return;
      stamps.push(meta.mediaTime);
      handle = video.requestVideoFrameCallback(collect);
    };
    handle = video.requestVideoFrameCallback(collect);
    await video.play();
    const began = performance.now();
    let sped = false;
    while (performance.now() - began < 2400 && stamps.length < 12 && !video.ended) {
      await new Promise((r) => setTimeout(r, 40));
      // コマ数の少ない動画（毎秒20コマ未満）は、ゆっくり再生だとコマが集まらないので途中から通常の速さにする。
      // それ以外は通常の速さにしない（処理の遅い端末でコマが飛ぶと、コマ数を少なく見積もってしまうため）
      if (!sped && performance.now() - began > 1300 && stamps.length < 6) {
        sped = true;
        video.playbackRate = 1;
      }
    }
  } catch {
    return null;
  } finally {
    running = false;
    try { video.cancelVideoFrameCallback?.(handle); } catch { /* 無視 */ }
    video.pause();
    video.playbackRate = prevRate;
    video.muted = wasMuted;
  }
  const diffs = [];
  for (let i = 1; i < stamps.length; i++) {
    const d = stamps[i] - stamps[i - 1];
    if (d > 1e-4) diffs.push(d);
  }
  if (diffs.length < 3) return null;
  diffs.sort((a, b) => a - b);
  // コマが飛ぶと 2コマ分・3コマ分の差が混ざる。本当の1コマの長さはいちばん小さい差なので、
  // 小さいほうから「ほぼ同じ値が2つ以上ある」ものを採用する
  let frame = 0;
  for (let i = 0; i < diffs.length - 1 && !frame; i++) {
    const group = diffs.filter((d) => d >= diffs[i] && d <= diffs[i] * 1.15);
    if (group.length >= 2) frame = group.reduce((a, b) => a + b, 0) / group.length;
  }
  if (!frame) return null;
  // 観測した区間全体で割り直して、1コマの長さを精密にする（コマ飛びがあっても正しく数えられる）
  const span = stamps[stamps.length - 1] - stamps[0];
  const count = Math.round(span / frame);
  if (count >= 2) frame = span / count;
  let fps = 1 / frame;
  if (!(fps >= 5 && fps <= 1000)) return null;
  // よくあるコマ数とほぼ一致していれば、その値にそろえる（60 と 59.94 は区別する）
  const near = COMMON_RATES.reduce((best, r) => (Math.abs(r - fps) < Math.abs(best - fps) ? r : best), COMMON_RATES[0]);
  if (Math.abs(near - fps) / near < 0.0004) { fps = near; frame = 1 / near; }
  return { fps, frame, phase: stamps[0] };
}

/**
 * 解析するコマの間隔を決める。
 * @param {{fps:number, frame:number, phase:number}|null} native 検出した動画のコマ数（分からなければ null）
 * @param {'auto'|'all'|'half'} mode auto: 動画に合わせる（最大60コマ/秒）、all: すべてのコマ、half: 速い（最大30コマ/秒・半分以下）
 */
export function samplingPlan(native, mode = 'auto') {
  const base = native || { fps: 30, frame: 1 / 30, phase: null };
  let step = 1;
  if (mode === 'auto') step = Math.max(1, Math.ceil(base.fps / 60 - 0.02));
  else if (mode === 'half') step = Math.max(2, Math.ceil(base.fps / 30 - 0.02));
  return { fps: base.fps / step, period: base.frame * step, frame: base.frame, phase: base.phase, step, native: !!native };
}

export class PoseExtractor {
  constructor() {
    this.landmarker = null;
    this.key = '';
    this.pending = null;
    this.pendingKey = '';
    this.requestId = 0;
    this.forceCpu = false;
    this.delegate = '';
    this.lastTimestamp = 0;
  }

  resolveKey({ model = 'full', delegate = 'auto' } = {}) {
    const wanted = delegate === 'auto' ? (this.forceCpu || prefersCpu() ? 'CPU' : 'GPU') : delegate;
    return { key: `${model}:${wanted}`, wanted };
  }

  /**
   * AIモデルを読み込む（同じ設定なら使い回す）。最後に頼まれた設定だけが有効になる。
   * @param {{model: 'lite'|'full'|'heavy', delegate?: 'auto'|'GPU'|'CPU'}} opts
   */
  async prepare({ model = 'full', delegate = 'auto' } = {}) {
    const { key, wanted } = this.resolveKey({ model, delegate });
    if (this.landmarker && this.key === key) return this.landmarker;
    if (this.pending && this.pendingKey === key) return this.pending;

    const id = ++this.requestId;
    const task = (async () => {
      const { FilesetResolver, PoseLandmarker } = await loadVision();
      const fileset = WASM_SIMD_ONLY
        ? { wasmLoaderPath: `${MEDIAPIPE_WASM}/vision_wasm_internal.js`, wasmBinaryPath: `${MEDIAPIPE_WASM}/vision_wasm_internal.wasm` }
        : await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM);
      const buffer = MODEL_FORMAT === 'base64' ? await loadModelBuffer(model) : null;
      const create = (d) => PoseLandmarker.createFromOptions(fileset, {
        baseOptions: buffer
          ? { modelAssetBuffer: buffer.slice(), delegate: d }
          : { modelAssetPath: MODEL_URLS[model], delegate: d },
        runningMode: 'VIDEO',
        numPoses: 1,
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
        outputSegmentationMasks: false,
      });
      let lm;
      let used = wanted;
      try {
        lm = await create(wanted);
      } catch (err) {
        if (wanted !== 'GPU') throw err;
        console.warn('GPUで初期化できなかったためCPUに切り替えます', err);
        used = 'CPU';
        lm = await create('CPU');
      }
      if (id !== this.requestId) {
        // 途中で別の設定が頼まれたので、これは使わない
        try { lm.close(); } catch { /* 無視 */ }
        return null;
      }
      if (this.landmarker) {
        try { this.landmarker.close(); } catch { /* 無視 */ }
      }
      this.landmarker = lm;
      this.key = key;
      this.delegate = used;
      this.lastTimestamp = 0;
      return lm;
    })();

    this.pending = task;
    this.pendingKey = key;
    try {
      return await task;
    } finally {
      if (this.pending === task) {
        this.pending = null;
        this.pendingKey = '';
      }
    }
  }

  /**
   * 動画の指定範囲を一定間隔で解析する
   * @param {HTMLVideoElement} video
   * @param {{start:number, end:number, plan:object, crop?:{x0:number,y0:number,x1:number,y1:number}|null,
   *   signal?:AbortSignal, onFrame?:Function}} opts
   *   plan: samplingPlan() の結果。コマの切れ目が分かっているときは、各コマの真ん中の時刻を取る
   *   crop: 人物のいる範囲（動画全体を0〜1とした座標）。指定するとその範囲だけを切り抜いて推定する
   */
  async extract(video, { start, end, plan, crop = null, signal, onFrame }) {
    if (!this.landmarker) throw new Error('モデルが読み込まれていません');
    const lm = this.landmarker;
    // 前の動画の追跡状態を引き継がないように、グラフを作り直す
    try { await lm.setOptions({ runningMode: 'VIDEO' }); } catch (err) { console.warn(err); }

    const { period, frame } = plan;
    // 最初のコマの時刻。コマの切れ目（phase）が分かっていれば、開始位置以降で最初のコマの真ん中にする
    let first = start + frame / 2;
    if (plan.phase != null) {
      const k0 = Math.ceil((start - plan.phase) / frame - 1e-6);
      first = plan.phase + (k0 + 0.5) * frame;
    }
    const last = Math.min(end, (Number.isFinite(video.duration) ? video.duration : end) - frame * 0.25);
    const count = Math.max(1, Math.floor((last - first) / period + 1e-6) + 1);
    const frames = new Array(count);
    // VIDEOモードではタイムスタンプが単調増加である必要がある
    const base = this.lastTimestamp + 1000;
    video.pause();

    // 範囲指定があるときは、その部分だけを canvas に描いて推定する
    let source = video;
    let map = null;
    let ctx = null;
    let rect = null;
    if (crop) {
      const W = video.videoWidth, H = video.videoHeight;
      const sx = Math.max(0, Math.floor(crop.x0 * W)), sy = Math.max(0, Math.floor(crop.y0 * H));
      const sw = Math.min(W - sx, Math.ceil((crop.x1 - crop.x0) * W)), sh = Math.min(H - sy, Math.ceil((crop.y1 - crop.y0) * H));
      if (sw >= 16 && sh >= 16) {
        const k = Math.min(1, 1280 / Math.max(sw, sh));
        const canvas = document.createElement('canvas');
        canvas.width = Math.max(16, Math.round(sw * k));
        canvas.height = Math.max(16, Math.round(sh * k));
        ctx = canvas.getContext('2d', { alpha: false });
        rect = { sx, sy, sw, sh };
        map = { ox: sx / W, oy: sy / H, kx: sw / W, ky: sh / H };
        source = canvas;
      }
    }

    let errors = 0;
    for (let i = 0; i < count; i++) {
      if (signal?.aborted) throw new DOMException('中止しました', 'AbortError');
      const t = first + i * period;
      await seekTo(video, t);
      if (signal?.aborted) throw new DOMException('中止しました', 'AbortError');
      let ts = base + Math.round((t - first) * 1000);
      if (ts <= this.lastTimestamp) ts = this.lastTimestamp + 1;
      let result = null;
      try {
        if (ctx) ctx.drawImage(video, rect.sx, rect.sy, rect.sw, rect.sh, 0, 0, source.width, source.height);
        result = lm.detectForVideo(source, ts);
      } catch (err) {
        console.warn('推定に失敗したコマ', t, err);
        errors++;
        if (errors === i + 1 && errors >= 5) {
          // 最初から失敗し続ける場合は処理方式の問題とみなす
          const e = new Error('姿勢の推定を実行できませんでした');
          e.code = 'DETECT_FAILED';
          throw e;
        }
      }
      this.lastTimestamp = ts;
      frames[i] = toFrame(result, t, map);
      onFrame?.(i, count, frames[i]);
    }

    return {
      fps: 1 / period,
      start: first,
      end: first + (count - 1) * period,
      frames,
      videoWidth: video.videoWidth,
      videoHeight: video.videoHeight,
    };
  }
}
