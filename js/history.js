// 解析結果の履歴。この端末のブラウザの中（IndexedDB）だけに保存し、どこにも送信しない。
//   takes : 一覧に出す情報（名前・日時・サムネイル・設定など）
//   data  : 解析データ（JSON文字列）
//   videos: 元の動画（任意）
const DB_NAME = 'form3d';
const DB_VERSION = 1;

let dbPromise = null;

function openDb() {
  if (!dbPromise) {
    dbPromise = new Promise((resolve, reject) => {
      if (typeof indexedDB === 'undefined') { reject(new Error('IndexedDB が使えません')); return; }
      let req;
      try {
        req = indexedDB.open(DB_NAME, DB_VERSION);
      } catch (err) {
        reject(err);
        return;
      }
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains('takes')) db.createObjectStore('takes', { keyPath: 'id', autoIncrement: true });
        if (!db.objectStoreNames.contains('data')) db.createObjectStore('data');
        if (!db.objectStoreNames.contains('videos')) db.createObjectStore('videos');
      };
      req.onsuccess = () => {
        const db = req.result;
        db.onversionchange = () => { db.close(); dbPromise = null; };
        resolve(db);
      };
      req.onerror = () => reject(req.error || new Error('履歴を開けませんでした'));
      req.onblocked = () => reject(new Error('履歴を開けませんでした'));
    }).catch((err) => {
      dbPromise = null;
      throw err;
    });
  }
  return dbPromise;
}

const wrap = (request) => new Promise((resolve, reject) => {
  request.onsuccess = () => resolve(request.result);
  request.onerror = () => reject(request.error);
});

const finished = (tx) => new Promise((resolve, reject) => {
  tx.oncomplete = () => resolve();
  tx.onerror = () => reject(tx.error);
  tx.onabort = () => reject(tx.error || new Error('保存できませんでした'));
});

/** 履歴が使える環境か（プライベートブラウズなどでは使えないことがある） */
export async function available() {
  try {
    await openDb();
    return true;
  } catch {
    return false;
  }
}

/**
 * 履歴に追加する。動画が大きすぎて保存できなかった場合は、解析データだけを残す。
 * @returns {Promise<{id:number, videoSaved:boolean}>}
 */
export async function addTake(meta, json, video = null) {
  const db = await openDb();
  const tx = db.transaction(['takes', 'data'], 'readwrite');
  const idReq = tx.objectStore('takes').add({ ...meta, hasVideo: false, videoSize: 0 });
  let id = null;
  idReq.onsuccess = () => {
    id = idReq.result;
    tx.objectStore('data').put(json, id);
  };
  await finished(tx);

  let videoSaved = false;
  if (video) {
    try {
      const vt = db.transaction(['videos', 'takes'], 'readwrite');
      vt.objectStore('videos').put(video, id);
      const takes = vt.objectStore('takes');
      const getReq = takes.get(id);
      getReq.onsuccess = () => {
        if (getReq.result) takes.put({ ...getReq.result, hasVideo: true, videoSize: video.size || 0 });
      };
      await finished(vt);
      videoSaved = true;
    } catch (err) {
      console.warn('動画を履歴に保存できませんでした', err);
    }
  }
  return { id, videoSaved };
}

/** 一覧用の情報を書き換える（名前や設定など） */
export async function updateTake(id, patch) {
  const db = await openDb();
  const tx = db.transaction('takes', 'readwrite');
  const store = tx.objectStore('takes');
  const req = store.get(id);
  req.onsuccess = () => {
    if (req.result) store.put({ ...req.result, ...patch, id });
  };
  await finished(tx);
}

/** 新しい順の一覧 */
export async function listTakes() {
  const db = await openDb();
  const all = await wrap(db.transaction('takes', 'readonly').objectStore('takes').getAll());
  return all.sort((a, b) => (b.createdAt || 0) - (a.createdAt || 0));
}

/** @returns {Promise<{meta:object, json:string, video:Blob|null}|null>} */
export async function getTake(id) {
  const db = await openDb();
  const tx = db.transaction(['takes', 'data', 'videos'], 'readonly');
  const [meta, json, video] = await Promise.all([
    wrap(tx.objectStore('takes').get(id)),
    wrap(tx.objectStore('data').get(id)),
    wrap(tx.objectStore('videos').get(id)),
  ]);
  if (!meta || typeof json !== 'string') return null;
  return { meta, json, video: video || null };
}

export async function deleteTake(id) {
  const db = await openDb();
  const tx = db.transaction(['takes', 'data', 'videos'], 'readwrite');
  tx.objectStore('takes').delete(id);
  tx.objectStore('data').delete(id);
  tx.objectStore('videos').delete(id);
  await finished(tx);
}

export async function clearAll() {
  const db = await openDb();
  const tx = db.transaction(['takes', 'data', 'videos'], 'readwrite');
  tx.objectStore('takes').clear();
  tx.objectStore('data').clear();
  tx.objectStore('videos').clear();
  await finished(tx);
}

/** このサイトが使っている保存容量（バイト）。分からなければ null */
export async function usageBytes() {
  try {
    const est = await navigator.storage?.estimate?.();
    return typeof est?.usage === 'number' ? est.usage : null;
  } catch {
    return null;
  }
}

/** ブラウザが勝手に消さないように頼む（断られても問題ない） */
export function requestPersistence() {
  try {
    navigator.storage?.persist?.().catch(() => {});
  } catch { /* 無視 */ }
}
