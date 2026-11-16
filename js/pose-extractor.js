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

/**
 * 人物の範囲（窓）。指定された枠の中から人物を探し、見つかったあとはその人物を追いかける。
 * 窓はいつも正方形で、中身を一定の大きさの canvas に描いて推定に使う。
 *   ・見つかる前: 利用者が囲んだ枠の中だけを描く（枠の外は黒く塗り、ほかの人を拾わないようにする）
 *   ・見つかった後: 人物を中心に、体の大きさに合わせた窓にする。近づいたり遠ざかったりしても追いかける
 *   ・見失ったとき: 窓を少しずつ広げて探し直す
 */
class PersonWindow {
  constructor(crop, W, H, follow, fps = 30) {
    this.W = W; this.H = H;
    this.follow = follow;
    this.maxLost = Math.max(30, Math.round(fps * 5)); // これだけ見つからなければ、最初の枠から探し直す
    this.box = { x: crop.x0 * W, y: crop.y0 * H, w: (crop.x1 - crop.x0) * W, h: (crop.y1 - crop.y0) * H };
    this.cx = this.box.x + this.box.w / 2;
    this.cy = this.box.y + this.box.h / 2;
    this.side = Math.max(this.box.w, this.box.h, 16);
    this.locked = false;
    this.lost = 0;
    this.sideAtLoss = this.side;
    this.ref = null; // 追いかけている人物の、直前の位置と大きさ { x0, y0, x1, y1, cx, cy, h }
    this.frameNo = 0;
    this.hidden = []; // 別人とみなして、しばらく隠す場所 { x0, y0, x1, y1, until }
    const size = follow ? 768 : Math.max(256, Math.min(1280, Math.round(this.side)));
    this.canvas = document.createElement('canvas');
    this.canvas.width = this.canvas.height = size;
    this.ctx = this.canvas.getContext('2d', { alpha: false });
  }

  /** 今の窓（動画全体を0〜1とした座標）。表示用 */
  rect() {
    if (!this.locked) {
      const b = this.box;
      return { x0: b.x / this.W, y0: b.y / this.H, x1: (b.x + b.w) / this.W, y1: (b.y + b.h) / this.H };
    }
    const h = this.side / 2;
    return {
      x0: Math.max(0, (this.cx - h) / this.W), y0: Math.max(0, (this.cy - h) / this.H),
      x1: Math.min(1, (this.cx + h) / this.W), y1: Math.min(1, (this.cy + h) / this.H),
    };
  }

  /** 窓の中身を canvas に描き、推定結果を動画全体の座標に戻すための係数を返す */
  draw(video) {
    const { ctx, canvas, W, H } = this;
    const S = canvas.width;
    const sx = this.cx - this.side / 2, sy = this.cy - this.side / 2, k = S / this.side;
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, S, S);
    // 描く範囲: 窓と動画の重なり。見つかる前は、さらに利用者の枠の中だけにする
    let x0 = Math.max(0, sx), y0 = Math.max(0, sy), x1 = Math.min(W, sx + this.side), y1 = Math.min(H, sy + this.side);
    if (!this.locked) {
      const b = this.box;
      x0 = Math.max(x0, b.x); y0 = Math.max(y0, b.y); x1 = Math.min(x1, b.x + b.w); y1 = Math.min(y1, b.y + b.h);
    }
    if (x1 - x0 >= 2 && y1 - y0 >= 2) {
      ctx.drawImage(video, x0, y0, x1 - x0, y1 - y0, (x0 - sx) * k, (y0 - sy) * k, (x1 - x0) * k, (y1 - y0) * k);
    }
    // 別人とみなした場所は黒く塗って、検出されないようにする
    this.frameNo++;
    this.hidden = this.hidden.filter((r) => r.until >= this.frameNo);
    for (const r of this.hidden) ctx.fillRect((r.x0 - sx) * k, (r.y0 - sy) * k, (r.x1 - r.x0) * k, (r.y1 - r.y0) * k);
    return { ox: sx / W, oy: sy / H, kx: this.side / W, ky: this.side / H };
  }

  // 推定結果の人物を囲む四角（画素）
  bounds(frame) {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (let j = 0; j < N; j++) {
      const x = frame.image[j * 3] * this.W, y = frame.image[j * 3 + 1] * this.H;
      if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
    }
    return { x0, y0, x1, y1, cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, h: y1 - y0 };
  }

  /**
   * 推定された人物が、追いかけている人物と同じ人かどうか。
   * 体の大きさや位置が急に変わったら別人とみなし、その場所をしばらく隠して探し直す。
   */
  accept(frame) {
    if (!this.follow || !this.locked || !this.ref) return true;
    const b = this.bounds(frame);
    const r = this.ref;
    const gap = this.lost + 1; // 最後に見えてから何コマ目か
    // 大きさ: 見失っている間も、同じ人なら映る大きさはあまり変わらない（少しだけゆるめる）
    const slack = Math.min(1.25, 1 + 0.01 * (gap - 1));
    const ratio = b.h / r.h;
    const sizeOk = ratio > 0.5 / slack && ratio < 2 * slack;
    // 位置: 見失っている時間が長いほど、離れた場所でも同じ人として受け入れる
    const distOk = Math.hypot(b.cx - r.cx, b.cy - r.cy) < r.h * (0.45 + 0.1 * gap);
    if (sizeOk && distOk) return true;
    // 追いかけている人物と大きく重なっていなければ、その場所を隠す
    const ix = Math.max(0, Math.min(b.x1, r.x1) - Math.max(b.x0, r.x0));
    const iy = Math.max(0, Math.min(b.y1, r.y1) - Math.max(b.y0, r.y0));
    const area = Math.max(1, (b.x1 - b.x0) * (b.y1 - b.y0));
    if ((ix * iy) / area < 0.4) {
      const mx = (b.x1 - b.x0) * 0.08, my = (b.y1 - b.y0) * 0.08;
      this.hidden.push({ x0: b.x0 - mx, y0: b.y0 - my, x1: b.x1 + mx, y1: b.y1 + my, until: this.frameNo + 45 });
    }
    return false;
  }

  /** 推定結果（動画全体の座標）を見て、次のコマの窓を決める */
  update(frame) {
    if (!this.follow) return; // 追いかけない設定: 最初の枠のまま
    if (!frame.image) {
      if (!this.locked) return;
      // 見失った: しばらくは同じ場所で待ち、それでも見つからなければ少しずつ広げて探す
      if (this.lost === 0) this.sideAtLoss = this.side;
      this.lost++;
      if (this.lost > 8) this.side = Math.min(this.side * 1.05, this.sideAtLoss * 2, Math.max(this.W, this.H) * 1.5);
      if (this.lost > this.maxLost) {
        // 長いあいだ見つからない: 最初に囲んだ枠に戻って、探し直す
        this.locked = false;
        this.ref = null;
        this.hidden = [];
        this.lost = 0;
        this.cx = this.box.x + this.box.w / 2;
        this.cy = this.box.y + this.box.h / 2;
        this.side = Math.max(this.box.w, this.box.h, 16);
      }
      return;
    }
    const b = this.bounds(frame);
    const { x0, y0, x1, y1 } = b;
    // 体の大きさは、急な変化（しゃがむ・腕を上げる）に引きずられないようにならして覚える
    b.h = this.ref ? this.ref.h + (b.h - this.ref.h) * 0.3 : b.h;
    this.ref = b;
    const want = Math.max(48, Math.min(Math.max(this.W, this.H) * 1.5, Math.max((y1 - y0) * 1.6, (x1 - x0) * 1.6)));
    const mx = (x0 + x1) / 2, my = (y0 + y1) / 2;
    if (!this.locked) {
      // 最初に見つけたとき: その人物に合わせた窓に切り替える
      this.locked = true;
      this.cx = mx; this.cy = my; this.side = want;
    } else {
      // 大きくなるほうにはすぐ、小さくなるほうにはゆっくり合わせる（しゃがんだ直後に立ち上がってもはみ出さない）
      const rate = want > this.side ? 0.5 : (this.side > want * 1.3 ? 0.25 : 0.08);
      this.side += (want - this.side) * rate;
      this.cx += (mx - this.cx) * 0.6;
      this.cy += (my - this.cy) * 0.6;
    }
    this.lost = 0;
  }
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
   *   follow?:boolean, signal?:AbortSignal, onFrame?:Function}} opts
   *   plan: samplingPlan() の結果。コマの切れ目が分かっているときは、各コマの真ん中の時刻を取る
   *   crop: 人物のいる範囲（動画全体を0〜1とした座標）。指定するとその中から人物を探す
   *   follow: 見つけた人物を追いかけるか（false なら最初の枠のまま）
   */
  async extract(video, { start, end, plan, crop = null, follow = true, signal, onFrame }) {
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
    const valid = crop && (crop.x1 - crop.x0) * video.videoWidth >= 16 && (crop.y1 - crop.y0) * video.videoHeight >= 16;
    const win = valid ? new PersonWindow(crop, video.videoWidth, video.videoHeight, follow, 1 / period) : null;

    let errors = 0;
    for (let i = 0; i < count; i++) {
      if (signal?.aborted) throw new DOMException('中止しました', 'AbortError');
      const t = first + i * period;
      await seekTo(video, t);
      if (signal?.aborted) throw new DOMException('中止しました', 'AbortError');
      let ts = base + Math.round((t - first) * 1000);
      if (ts <= this.lastTimestamp) ts = this.lastTimestamp + 1;
      let result = null;
      let map = null;
      const shown = win ? win.rect() : null;
      try {
        if (win) map = win.draw(video);
        result = lm.detectForVideo(win ? win.canvas : video, ts);
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
      if (win) {
        // 追いかけている人物と別の人を拾ったコマは、検出なしとして扱う
        if (frames[i].image && !win.accept(frames[i])) frames[i] = { t, world: null, image: null, vis: null };
        win.update(frames[i]);
        frames[i].win = shown; // このコマで使った窓（解析中の表示用）
      }
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
