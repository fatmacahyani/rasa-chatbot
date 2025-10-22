// server.js
const express = require("express");
const cors = require("cors");
const db = require("./database"); // MySQL connection/pool

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

// =======================================
// Table names (adjust if needed)
// =======================================
const TABLE_BIAYA = "t_biaya";
const TABLE_PRODI = "t_prodi";

// =======================================
// DB helper (promise wrapper)
// =======================================
function q(sql, params = []) {
  return new Promise((resolve, reject) => {
    db.query(sql, params, (err, rows) => {
      if (err) return reject(err);
      resolve(rows);
    });
  });
}

// =======================================
// Fakultas alias (short code -> full name)
// =======================================
const FAKULTAS_ALIAS = {
  ELECTICS: "Teknologi Elektro dan Informatika Cerdas",
  INDSYS: "Teknologi Industri dan Rekayasa Sistem",
  CIVPLAN: "Teknik Sipil, Perencanaan, dan Kebumian",
  MARTECH: "Teknologi Kelautan",
  SCIENTICS: "Sains dan Analitika Data",
  SIMT: "Sekolah Interdisiplin Manajemen dan Teknologi",
  CREABIZ: "Fakultas Desain Kreatif dan Bisnis Digital",
};

// Simple normalizer
function norm(x) {
  if (!x) return x;
  return String(x).trim();
}

// =======================================
// Caches (for robust NLU on prodi/fakultas)
// =======================================
let PRODI_CACHE = [];    // e.g., ["Teknik Informatika", ...]
let FAKULTAS_CACHE = []; // e.g., ["Teknologi Elektro dan Informatika Cerdas", ...]

function buildFakultasLookup() {
  const fullNames = Object.values(FAKULTAS_ALIAS);
  FAKULTAS_CACHE = Array.from(new Set(fullNames));
}

async function refreshCache() {
  try {
    const rows = await q(`SELECT fakultas, prodi FROM ${TABLE_PRODI} ORDER BY fakultas, prodi`);
    PRODI_CACHE = rows.map((r) => r.prodi).filter(Boolean);
    buildFakultasLookup();
    console.log(`  Cache loaded: ${PRODI_CACHE.length} prodi, ${FAKULTAS_CACHE.length} fakultas`);
  } catch (e) {
    console.error("Cache load error:", e);
  }
}
// load once at startup
refreshCache();

// =======================================
// NLU: ONLY prodi <-> fakultas
// (No static QnA here, by request)
// =======================================
function detectFakultas(text) {
  const s = (text || "").toLowerCase();

  // 1) alias codes (ELECTICS, INDSYS, ...)
  for (const [alias, full] of Object.entries(FAKULTAS_ALIAS)) {
    const re = new RegExp(`\\b${alias.toLowerCase()}\\b`, "i");
    if (re.test(s)) return full;
  }

  // 2) full names from cache
  for (const f of FAKULTAS_CACHE) {
    const re = new RegExp(f.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "i");
    if (re.test(s)) return f;
  }

  // 3) simple heuristics
  if (/kelautan/i.test(s)) return "Teknologi Kelautan";
  if (/elektro|informatika/i.test(s)) return "Teknologi Elektro dan Informatika Cerdas";
  if (/industri|rekayasa sistem/i.test(s)) return "Teknologi Industri dan Rekayasa Sistem";
  if (/sipil|perencanaan|kebumian/i.test(s)) return "Teknik Sipil, Perencanaan, dan Kebumian";
  if (/sains|analitika|scientics/i.test(s)) return "Sains dan Analitika Data";
  if (/interdisiplin|manajemen teknologi|simt/i.test(s)) return "Sekolah Interdisiplin Manajemen dan Teknologi";
  if (/desain kreatif|bisnis digital|creabiz/i.test(s)) return "Fakultas Desain Kreatif dan Bisnis Digital";

  return null;
}

function detectProdi(text) {
  const s = (text || "").toLowerCase();

  // 1) exact/partial from DB cache
  for (const p of PRODI_CACHE) {
    const re = new RegExp(p.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "i");
    if (re.test(s)) return p;
  }

  // 2) common synonyms (fallbacks)
  const synonyms = [
    { re: /(informatika|computer science|cs)/i, val: "Teknik Informatika" },
    { re: /(teknik elektro|electrical)/i, val: "Teknik Elektro" },
    { re: /(teknik sipil)/i, val: "Teknik Sipil" },
    { re: /(manajemen teknologi)/i, val: "Manajemen Teknologi" },
    { re: /(statistika|statistik)/i, val: "Statistika" },
    { re: /(arsitektur)/i, val: "Arsitektur" },
  ];
  for (const syn of synonyms) {
    if (syn.re.test(s)) return syn.val;
  }

  return null;
}

function classifyIntentForProdiFakultas(text, entities) {
  const s = (text || "").toLowerCase();
  const hasKeywordProdi = /(prodi|program studi|jurusan)/i.test(s);
  const hasKeywordFakultas = /fakultas/i.test(s);
  const askFacultyOfProdi = /(masuk fakultas apa|fakultas apa|di fakultas mana)/i.test(s);

  const hasFakEnt = entities.some((e) => e.entity === "fakultas");
  const hasProdiEnt = entities.some((e) => e.entity === "prodi");

  // examples: "prodi di ELECTICS", "daftar prodi fakultas teknologi kelautan"
  if (hasKeywordProdi && (hasFakEnt || /di\s+\w+/i.test(s) || hasKeywordFakultas)) {
    return { name: "tanya_prodi_tersedia", confidence: 0.93 };
  }

  // examples: "teknik informatika masuk fakultas apa", "fakultas apa untuk prodi ..."
  if (askFacultyOfProdi || (hasKeywordFakultas && hasProdiEnt)) {
    return { name: "cari_fakultas_dari_prodi", confidence: 0.93 };
  }

  // default fallback
  return { name: "nlu_fallback", confidence: 0.1 };
}

// Endpoint used by Rasa NLUCommandAdapter
app.post("/parse", async (req, res) => {
  try {
    const { text } = req.body || {};
    const entities = [];

    const f = detectFakultas(text);
    if (f) entities.push({ entity: "fakultas", value: f, start: 0, end: 0 });

    const p = detectProdi(text);
    if (p) entities.push({ entity: "prodi", value: p, start: 0, end: 0 });

    const intent = classifyIntentForProdiFakultas(text, entities);

    console.log(
      ` /parse "${text}" -> ${intent.name} (${intent.confidence}) | ents=${entities
        .map((e) => `${e.entity}:${e.value}`)
        .join(", ")}`
    );

    res.json({ intent, entities, text });
  } catch (e) {
    console.error("NLU /parse error:", e);
    res.status(200).json({
      intent: { name: "nlu_fallback", confidence: 0.1 },
      entities: [],
      text: req.body?.text || "",
    });
  }
});

// =======================================
// Health
// =======================================
app.get("/health", (req, res) => {
  res.json({
    status: "OK",
    message: "Pasca Data API running",
    db: "info_pasca",
    endpoints: ["/parse", "/biaya", "/prodi", "/fakultas"],
    ts: new Date().toISOString(),
  });
});

// =======================================
// GET /biaya
// ?jenjang=Magister&prodi=Informatika  (prodi|program optional)
// ?limit=10
// =======================================
app.get("/biaya", async (req, res) => {
  try {
    let { jenjang, prodi, program, limit } = req.query;
    jenjang = norm(jenjang);
    const likeVal = norm(prodi || program || "");
    limit = Number(limit) > 0 ? Number(limit) : null;

    let sql = `
      SELECT program, jenjang, biaya_label, spi, ipits
      FROM ${TABLE_BIAYA}
      WHERE 1=1
    `;
    const params = [];

    if (jenjang) {
      sql += ` AND LOWER(jenjang) LIKE LOWER(?)`;
      params.push(`%${jenjang}%`);
    }
    if (likeVal) {
      sql += ` AND LOWER(program) LIKE LOWER(?)`;
      params.push(`%${likeVal}%`);
    }

    sql += ` ORDER BY jenjang, program`;
    if (limit) sql += ` LIMIT ${limit}`;

    let rows = await q(sql, params);

    // fallback to jenjang-only if user provided program/prodi but got 0 rows
    if ((!rows || rows.length === 0) && likeVal && jenjang) {
      let sql2 = `
        SELECT program, jenjang, biaya_label, spi, ipits
        FROM ${TABLE_BIAYA}
        WHERE LOWER(jenjang) LIKE LOWER(?)
        ORDER BY jenjang, program
      `;
      if (limit) sql2 += ` LIMIT ${limit}`;
      rows = await q(sql2, [`%${jenjang}%`]);
      console.log(`[biaya] fallback to jenjang-only: ${jenjang} -> ${rows.length} rows`);
    }

    res.json(rows);
  } catch (err) {
    console.error(" /biaya error:", err);
    res.status(500).json({ error: "Database error", detail: err.message });
  }
});

// =======================================
// GET /prodi
// ?fakultas=ELECTICS | full name | partial
// ?limit=50
// =======================================
app.get("/prodi", async (req, res) => {
  try {
    let { fakultas, limit } = req.query;
    limit = Number(limit) > 0 ? Number(limit) : null;

    // alias -> full name
    if (fakultas && FAKULTAS_ALIAS[fakultas.toUpperCase()]) {
      fakultas = FAKULTAS_ALIAS[fakultas.toUpperCase()];
    }
    fakultas = norm(fakultas);

    let sql = `SELECT fakultas, prodi FROM ${TABLE_PRODI} WHERE 1=1`;
    const params = [];
    if (fakultas) {
      sql += ` AND LOWER(fakultas) LIKE LOWER(?)`;
      params.push(`%${fakultas}%`);
    }
    sql += ` ORDER BY fakultas, prodi`;
    if (limit) sql += ` LIMIT ${limit}`;

    const rows = await q(sql, params);
    res.json(rows);
  } catch (err) {
    console.error(" /prodi error:", err);
    res.status(500).json({ error: "Database error", detail: err.message });
  }
});

// =======================================
// GET /fakultas
// ?prodi=Teknik Informatika
// =======================================
app.get("/fakultas", async (req, res) => {
  try {
    let { prodi } = req.query;
    if (!prodi) {
      return res.status(400).json({ error: "Query param 'prodi' is required" });
    }
    prodi = norm(prodi);

    const sql = `
      SELECT fakultas
      FROM ${TABLE_PRODI}
      WHERE LOWER(prodi) LIKE LOWER(?)
      LIMIT 1
    `;
    const rows = await q(sql, [`%${prodi}%`]);
    if (!rows.length) return res.json({});
    res.json({ fakultas: rows[0].fakultas });
  } catch (err) {
    console.error(" /fakultas error:", err);
    res.status(500).json({ error: "Database error", detail: err.message });
  }
});

// =======================================
// 404 & error handler
// =======================================
app.use((req, res) => res.status(404).json({ error: "Not Found" }));
app.use((err, req, res, next) => {
  console.error(" Uncaught error:", err);
  res.status(500).json({ error: "Internal Server Error" });
});

// =======================================
// Start server
// =======================================
app.listen(PORT, () => {
  console.log(` Pasca Data API listening on http://localhost:${PORT}`);
  console.log(`   • POST /parse   (NLU for prodi/fakultas)`);
  console.log(`   • GET  /biaya?jenjang=Magister&prodi=Informatika`);
  console.log(`   • GET  /prodi?fakultas=ELECTICS`);
  console.log(`   • GET  /fakultas?prodi=Teknik%20Informatika`);
});
