// フォーム3D — オフライン対応（サービスワーカー）
//
// ・このサイト自身のファイル: まずネットワークから取り、だめならキャッシュ（更新がすぐ反映される）
// ・バージョン固定のライブラリとAIモデル: 一度取ったらキャッシュから（中身が変わらないため）
// ・それ以外の通信（広告など）には一切さわらない
const APP_CACHE = 'form3d-app-v3';
const LIB_CACHE = 'form3d-lib-v1';

const LIB_PREFIXES = [
  'https://cdn.jsdelivr.net/npm/three@0.170.0/',
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/',
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/',
];

// 最初に入れておく、このサイトの基本ファイル
const APP_SHELL = [
  './',
  'css/style.css',
  'css/pages.css',
  'js/main.js',
  'js/config.js',
  'js/landmarks.js',
  'js/pose-extractor.js',
  'js/motion.js',
  'js/analysis.js',
  'js/chart.js',
  'js/overlay2d.js',
  'js/viewer3d.js',
  'js/io.js',
  'js/history.js',
  'guide.html',
  'metrics.html',
  'faq.html',
  'privacy.html',
  'about.html',
  'manifest.webmanifest',
  'icons/icon-192.png',
  'icons/apple-touch-icon.png',
];

// 3D表示のライブラリは、ページを開いた直後（このプログラムが動き出す前）に読み込まれるので、
// ここで先にキャッシュへ入れておく。姿勢推定のプログラムとAIモデルは大きいので、最初に使ったときに入る
const THREE_BASE = 'https://cdn.jsdelivr.net/npm/three@0.170.0/';
const LIB_PRECACHE = [
  'build/three.module.js',
  'examples/jsm/controls/OrbitControls.js',
  'examples/jsm/lines/Line2.js',
  'examples/jsm/lines/LineMaterial.js',
  'examples/jsm/lines/LineGeometry.js',
  'examples/jsm/lines/LineSegments2.js',
  'examples/jsm/lines/LineSegmentsGeometry.js',
].map((p) => THREE_BASE + p);

const SCOPE = new URL('./', self.location.href);

self.addEventListener('install', (event) => {
  event.waitUntil((async () => {
    const cache = await caches.open(APP_CACHE);
    // 1つ取れなくてもインストールは続ける
    await Promise.all(APP_SHELL.map((path) => cache.add(new Request(path, { cache: 'reload' })).catch(() => {})));
    const lib = await caches.open(LIB_CACHE);
    await Promise.all(LIB_PRECACHE.map(async (url) => {
      try {
        if (await lib.match(url)) return;
        const res = await fetch(url, { mode: 'cors', credentials: 'omit' });
        if (res.ok) await lib.put(url, res);
      } catch { /* 取れなくても続ける */ }
    }));
    await self.skipWaiting();
  })());
});

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(keys
      .filter((k) => k.startsWith('form3d-') && k !== APP_CACHE && k !== LIB_CACHE)
      .map((k) => caches.delete(k)));
    await self.clients.claim();
  })());
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;
  if (LIB_PREFIXES.some((p) => req.url.startsWith(p))) {
    event.respondWith(cacheFirst(req));
    return;
  }
  const url = new URL(req.url);
  if (url.origin === SCOPE.origin && url.pathname.startsWith(SCOPE.pathname) && !req.headers.has('range')) {
    event.respondWith(networkFirst(req));
  }
});

async function cacheFirst(req) {
  const cache = await caches.open(LIB_CACHE);
  const hit = await cache.match(req.url);
  if (hit) return hit;
  // <script> からの読み込みでも、中身を確認できる形（CORS）で取り直して保存する
  const res = await fetch(req.url, { mode: 'cors', credentials: 'omit' });
  if (res.ok) cache.put(req.url, res.clone()).catch(() => {});
  return res;
}

async function networkFirst(req) {
  const cache = await caches.open(APP_CACHE);
  const cached = () => cache.match(req, { ignoreSearch: true })
    .then((hit) => hit || (req.mode === 'navigate' ? cache.match('./', { ignoreSearch: true }) : undefined));

  const network = fetch(req).then((res) => {
    if (res.ok && res.type === 'basic') cache.put(req, res.clone()).catch(() => {});
    return res;
  });

  // 電波が悪くて4秒たっても返ってこないときは、キャッシュがあればそれを使う
  let timer;
  const slow = new Promise((resolve) => { timer = setTimeout(() => resolve(null), 4000); });
  try {
    const first = await Promise.race([network, slow]);
    if (first) return first;
    const hit = await cached();
    if (hit) { network.catch(() => {}); return hit; }
    return await network;
  } catch (err) {
    const hit = await cached();
    if (hit) return hit;
    throw err;
  } finally {
    clearTimeout(timer);
  }
}
