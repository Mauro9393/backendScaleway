// server.js
require("dotenv").config();
const express = require("express");
const axios = require("axios");
const cors = require("cors");
const { OpenAI } = require("openai");
const multer = require("multer");
const FormData = require("form-data");
const fs = require("fs");
const path = require("path");
const { VertexAI } = require("@google-cloud/vertexai");
const { Pool } = require('pg');

// Multer setup for file uploads
const upload = multer({ storage: multer.memoryStorage() });

const app = express();
const port = process.env.PORT || 3000;

// Write Google service account key to disk and set env var
const saKey = process.env.GOOGLE_SERVICE_ACCOUNT_KEY;
const keyPath = path.join("/tmp", "sa-key.json");
fs.writeFileSync(keyPath, saKey);
process.env.GOOGLE_APPLICATION_CREDENTIALS = keyPath;

// OpenAI SDK
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY_SIMULATEUR });

// Vertex AI setup
const vertexAI = new VertexAI({
    project: process.env.GCLOUD_PROJECT,
    location: process.env.VERTEX_LOCATION
});
const vertexModel = vertexAI.getGenerativeModel({
    model: process.env.VERTEX_MODEL_ID,
    generationConfig: { maxOutputTokens: 2048 }
});

// Axios with timeout (for Azure/OpenAI HTTP calls)
const API_TIMEOUT = 320000; // 5 minutes
const axiosInstance = axios.create({ timeout: API_TIMEOUT });

// Global middlewares
app.use(cors({
    origin: "*",
    methods: ["GET", "POST", "OPTIONS"],
    allowedHeaders: ["Content-Type"]
}));
app.use(express.json());

// pool DEVE stare prima di qualsiasi route che lo usa
const pool = new Pool({
    connectionString: process.env.PG_CONNECTION,
    ssl: { rejectUnauthorized: false }
});

// SSE helper for OpenAI Threads
async function streamAssistant(assistantId, messages, userId, res) {
    const thread = await openai.beta.threads.create({ messages });
    const run = await openai.beta.threads.runs.createAndStream(
        thread.id,
        { assistant_id: assistantId, stream: true, user: userId }
    );
    for await (const event of run) {
        const delta = event.data?.delta?.content;
        if (delta) res.write(`data: ${JSON.stringify({ delta })}\n\n`);
    }
    res.write("data: [DONE]\n\n");
    res.end();
}

// Whisper transcription endpoint
app.post("/api/transcribe", upload.single("audio"), async (req, res) => {
    console.log("🔹 /api/transcribe, req.file:", req.file?.originalname, req.file?.size);
    const apiKey = process.env.OPENAI_API_KEY_SIMULATEUR;
    if (!apiKey) return res.status(500).json({ error: "OpenAI API key missing" });
    if (!req.file) return res.status(400).json({ error: "No audio file uploaded" });
    try {
        const form = new FormData();
        form.append("file", req.file.buffer, { filename: req.file.originalname });
        form.append("model", "whisper-1");
        const response = await axios.post(
            "https://api.openai.com/v1/audio/transcriptions",
            form,
            { headers: { ...form.getHeaders(), Authorization: `Bearer ${apiKey}` } }
        );
        console.log("🎉 Whisper response:", response.data);
        return res.json(response.data);
    } catch (err) {
        const details = err.response?.data || err.message;
        console.error("❌ Whisper transcription error details:", details);
        return res.status(err.response?.status || 500)
            .json({ error: "Transcription failed", details });
    }
});

// Main API router
app.post("/api/:service", upload.none(), async (req, res) => {
    const { service } = req.params;
    console.log("🔹 Servizio ricevuto:", service);
    console.log("🔹 Dati ricevuti:", JSON.stringify(req.body));
    try {
        // Azure OpenAI Chat (Simulator)
        if (service === "azureOpenaiSimulateur") {
            const apiKey = process.env.AZURE_OPENAI_KEY_SIMULATEUR;
            const endpoint = process.env.AZURE_OPENAI_ENDPOINT_SIMULATEUR;
            const deployment = process.env.AZURE_OPENAI_DEPLOYMENT_SIMULATEUR;
            const apiVersion = process.env.AZURE_OPENAI_API_VERSION;
            const apiUrl = `${endpoint}/openai/deployments/${deployment}/chat/completions?api-version=${apiVersion}`;
            const response = await axiosInstance.post(apiUrl, req.body, {
                headers: { 'api-key': apiKey, 'Content-Type': 'application/json' },
                responseType: 'stream'
            });
            res.setHeader("Content-Type", "text/event-stream");
            res.setHeader("Cache-Control", "no-cache");
            response.data.on('data', chunk => res.write(chunk));
            response.data.on('end', () => res.end());
            return;
        }

        // Vertex Chat (batch + streaming)
        else if (service === "vertexChat") {
            // CORS already applied globally
            const { messages, stream = true } = req.body;
            const promptText = messages.map(m => `${m.role.toUpperCase()}: ${m.content}`).join("\n");
            const request = { contents: [{ role: "user", parts: [{ text: promptText }] }] };

            // Batch invocation
            if (stream === false) {
                try {
                    const result = await vertexModel.generateContent({
                        ...request,
                        generationConfig: { maxOutputTokens: 2048 }
                    });
                    const text = result.response?.candidates?.[0]?.content?.parts?.[0]?.text || "";
                    return res.json({ text });
                } catch (err) {
                    console.error("Vertex AI batch error:", err);
                    return res.status(500).json({ error: err.message });
                }
            }

            // Streaming invocation
            try {
                res.setHeader("Content-Type", "text/event-stream");
                res.setHeader("Cache-Control", "no-cache");
                res.flushHeaders();
                const result = await vertexModel.generateContentStream(request);
                for await (const item of result.stream) {
                    const delta = item.candidates?.[0]?.content?.parts?.[0]?.text;
                    if (delta) res.write(`data: ${JSON.stringify({ delta })}\n\n`);
                }
                res.write("data: [DONE]\n\n");
                return res.end();
            } catch (err) {
                console.error("Vertex AI streaming error:", err);
                if (!res.headersSent) {
                    return res.status(500).json({ error: err.message });
                }
                res.write(`data: ${JSON.stringify({ error: err.message })}\n\n`);
                res.write("data: [DONE]\n\n");
                return res.end();
            }
        }

        // OpenAI streaming (SDK)
        else if (service === "openaiSimulateur") {
            res.setHeader("Content-Type", "text/event-stream");
            res.setHeader("Cache-Control", "no-cache");
            res.flushHeaders();
            const stream = await openai.chat.completions.create({
                model: req.body.model,
                messages: req.body.messages,
                stream: true
            });
            for await (const part of stream) {
                const delta = part.choices?.[0]?.delta?.content;
                if (delta) {
                    res.write(`data: ${JSON.stringify({ choices: [{ delta: { content: delta } }] })}\n\n`);
                }
            }
            const totalTokens = stream.usage?.total_tokens || 0;
            res.write(`data: ${JSON.stringify({ usage: { total_tokens: totalTokens } })}\n\n`);
            res.write("data: [DONE]\n\n");
            return res.end();
        }

        // OpenAI Analyse (non-stream via Threads)
        else if (service === "assistantOpenaiAnalyse") {
            const assistantId = process.env.OPENAI_ASSISTANTID;
            // create thread & run, then poll until complete
            const thread = await openai.beta.threads.create({ messages: req.body.messages });
            const run = await openai.beta.threads.runs.create(thread.id, { assistant_id: assistantId });
            let status;
            do {
                await new Promise(r => setTimeout(r, 1000));
                status = await openai.beta.threads.runs.retrieve(thread.id, run.id);
            } while (status.status !== "completed" && status.status !== "failed");
            if (status.status !== "completed") {
                return res.status(500).json({ error: "Assistant run failed", details: status.status });
            }
            const msgs = await openai.beta.threads.messages.list(thread.id, { limit: 1, order: "desc" });
            const answer = msgs.data[0]?.content?.[0]?.text?.value || "";
            return res.json({ answer });
        }

        // Assistant OpenAI Analyse Streaming
        else if (service === "assistantOpenaiAnalyseStreaming") {
            const assistantId = process.env.OPENAI_ASSISTANTID;
            let thread;
            if (req.body.threadId) {
                thread = { id: req.body.threadId };
                for (const msg of req.body.messages) {
                    await openai.beta.threads.messages.create(thread.id, msg);
                }
            } else {
                thread = await openai.beta.threads.create({ messages: req.body.messages });
            }
            res.setHeader("Content-Type", "text/event-stream");
            res.setHeader("Cache-Control", "no-cache");
            res.flushHeaders();
            if (!req.body.threadId) {
                res.write(`data: ${JSON.stringify({ threadId: thread.id })}\n\n`);
            }
            const stream = await openai.beta.threads.runs.createAndStream(thread.id, { assistant_id: assistantId, stream: true });
            for await (const event of stream) {
                const delta = event.data?.delta?.content;
                if (delta) res.write(`data: ${JSON.stringify({ delta })}\n\n`);
            }
            res.write("data: [DONE]\n\n");
            return res.end();
        }

        // OpenAI Analyse (chat completions non-stream)
        else if (service === "openaiAnalyse") {
            res.setHeader("Content-Type", "text/event-stream");
            res.setHeader("Cache-Control", "no-cache");
            res.setHeader("Connection", "keep-alive");
            res.setHeader("Access-Control-Allow-Origin", "*");
            res.flushHeaders();
            try {
                const stream = await openai.chat.completions.create({
                    model: req.body.model,
                    messages: req.body.messages,
                    stream: true
                });
                for await (const part of stream) {
                    const delta = part.choices?.[0]?.delta?.content;
                    if (delta) res.write(`data: ${JSON.stringify({ delta })}\n\n`);
                }
                res.write("data: [DONE]\n\n");
                res.end();
            } catch (err) {
                console.error("❌ Errore nello stream openaiAnalyse:", err.message);
                res.write(`data: ${JSON.stringify({ error: "Errore durante lo streaming AI." })}\n\n`);
                res.end();
            }
        }

        // Azure OpenAI Analyse (batch)
        else if (service === "azureOpenaiAnalyse") {
            const apiKey = process.env.AZURE_OPENAI_KEY_SIMULATEUR;
            const endpoint = process.env.AZURE_OPENAI_ENDPOINT_SIMULATEUR;
            const deployment = process.env.AZURE_OPENAI_DEPLOYMENT_COACH;
            const apiVersion = process.env.AZURE_OPENAI_API_VERSION_COACH;
            const apiUrl = `${endpoint}/openai/deployments/${deployment}/chat/completions?api-version=${apiVersion}`;
            try {
                const response = await axiosInstance.post(apiUrl, req.body, { headers: { 'api-key': apiKey, 'Content-Type': 'application/json' } });
                return res.json(response.data);
            } catch (err) {
                console.error("❌ Azure Analyse Error:", err.response?.data || err.message);
                return res.status(err.response?.status || 500).json(err.response?.data || { error: "Errore interno Azure Analyse" });
            }
        }

        // OpenAI TTS (batch)
        else if (service === "openai-tts") {
            const apiKey = process.env.OPENAI_API_KEY_SIMULATEUR;
            if (!apiKey) return res.status(500).json({ error: "OpenAI API key missing" });
            const { text, selectedVoice } = req.body;
            if (!text) return res.status(400).json({ error: "Text is required" });
            const allowedVoices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"];
            const voice = allowedVoices.includes((selectedVoice || "").trim().toLowerCase())
                ? selectedVoice.trim().toLowerCase()
                : "fable";
            try {
                const response = await axios.post(
                    "https://api.openai.com/v1/audio/speech",
                    { model: "gpt-4o-mini-tts", input: text, voice, instructions: "Speak in a gentle, slow and friendly way." },
                    { headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" }, responseType: "arraybuffer" }
                );
                res.setHeader("Content-Type", "audio/mpeg");
                return res.send(response.data);
            } catch (err) {
                console.error("OpenAI TTS error:", err.response?.data || err.message);
                return res.status(err.response?.status || 500).json({ error: "OpenAI TTS failed", details: err.message });
            }
        }

        // OpenAI Streaming TTS (SDK)
        else if (service === "streaming-openai-tts") {
            const { text, selectedVoice } = req.body;
            if (!text) return res.status(400).json({ error: "Text is required" });
            const allowed = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"];
            const voice = allowed.includes((selectedVoice || "").trim().toLowerCase())
                ? selectedVoice.trim().toLowerCase()
                : "fable";
            try {
                const ttsResp = await openai.audio.speech.create({ model: "tts-1", input: text, voice, instructions: "Speak in a cheerful and positive tone.", response_format: "mp3" });
                res.setHeader("Content-Type", "audio/mpeg");
                res.setHeader("Transfer-Encoding", "chunked");
                ttsResp.body.pipe(res);
            } catch (err) {
                console.error("OpenAI TTS error:", err);
                return res.status(500).json({ error: "OpenAI TTS failed" });
            }
        }

        // Azure Text-to-Speech
        else if (service === "azureTextToSpeech") {
            const { text, selectedVoice } = req.body;
            if (!text) return res.status(400).json({ error: "Text is required" });
            const endpoint = process.env.AZURE_TTS_ENDPOINT;
            const apiKey = process.env.AZURE_TTS_KEY;
            const deployment = "tts";
            const apiVersion = "2025-03-01-preview";
            const url = `${endpoint}/openai/deployments/${deployment}/audio/speech?api-version=${apiVersion}`;
            const voiceMap = { alloy: "alloy", echo: "echo", fable: "fable", onyx: "onyx", nova: "nova", shimmer: "shimmer" };
            const voice = voiceMap[(selectedVoice || "").trim().toLowerCase()] || "fable";
            try {
                const response = await axios.post(url, { model: "tts-1", input: text, voice },
                    { headers: { "Content-Type": "application/json", "api-key": apiKey, "Accept": "audio/mpeg" }, responseType: "arraybuffer" }
                );
                res.setHeader("Content-Type", "audio/mpeg");
                return res.send(response.data);
            } catch (err) {
                console.error("Azure TTS error:", err.response?.data || err.message);
                return res.status(err.response?.status || 500).json({ error: "Azure TTS failed", details: err.message });
            }
        }
        else if (service === "userList") {
            const { clientName, userID, userName, userScore, historique, rapport } = req.body;
            try {
                const result = await pool.query(
                    "INSERT INTO userlist (client_name, user_id, name, score, historique, rapport) VALUES ($1, $2, $3, $4, $5, $6) RETURNING *",
                    [clientName, userID, userName, userScore, historique, rapport]
                );
                return res
                    .status(201)
                    .header("Access-Control-Allow-Origin", "*")
                    .header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
                    .header("Access-Control-Allow-Headers", "Content-Type")
                    .json({ message: "Utente inserito!", data: result.rows[0] });
            } catch (err) {
                console.error("❌ Errore inserimento userList:", err);
                // manda il messaggio d’errore al client per debug
                res
                    .status(500)
                    .header("Access-Control-Allow-Origin", "*")
                    .header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
                    .header("Access-Control-Allow-Headers", "Content-Type")
                    .json({ error: err.message });
            }
        }
        // ElevenLabs TTS
        else if (service === "elevenlabs") {
            const apiKey = process.env.ELEVENLAB_API_KEY;
            if (!apiKey) return res.status(500).json({ error: "ElevenLabs API key missing" });
            const { text, selectedLanguage } = req.body;
            const voiceMap = { espagnol: "l1zE9xgNpUTaQCZzpNJa", français: "1a3lMdKLUcfcMtvN772u", anglais: "7tRwuZTD1EWi6nydVerp" };
            const lang = (selectedLanguage || "").trim().toLowerCase();
            const voiceId = voiceMap[lang];
            if (!voiceId) return res.status(400).json({ error: "Not supported language" });
            const apiUrl = `https://api.elevenlabs.io/v1/text-to-speech/${voiceId}/stream`;
            try {
                const response = await axios.post(apiUrl,
                    { text, model_id: "eleven_flash_v2_5", voice_settings: { stability: 0.6, similarity_boost: 0.7, style: 0.1 } },
                    { headers: { "xi-api-key": apiKey, "Content-Type": "application/json" }, responseType: "arraybuffer" }
                );
                console.log("Audio received from ElevenLabs!");
                res.setHeader("Content-Type", "audio/mpeg");
                return res.send(response.data);
            } catch (err) {
                if (err.response) {
                    let msg;
                    try { msg = err.response.data.toString(); } catch { msg = "Unknown error"; }
                    console.error("ElevenLabs error:", msg);
                    return res.status(err.response.status).json({ error: msg });
                }
                console.error("Unknown error ElevenLabs:", err.message);
                return res.status(500).json({ error: "Unknown error with ElevenLabs" });
            }
        }

        // Fallback invalid service
        else {
            return res.status(400).json({ error: "Invalid service" });
        }
    } catch (error) {
        console.error(`API error ${req.params.service}:`, error.response?.data || error.message);
        res.status(500).json({ error: "API request error" });
    }
});

// Secure endpoint to obtain Azure Speech token
app.get("/get-azure-token", async (req, res) => {
    const apiKey = process.env.AZURE_SPEECH_API_KEY;
    const region = process.env.AZURE_REGION;
    if (!apiKey || !region) return res.status(500).json({ error: "Azure keys missing in the backend" });
    try {
        const tokenRes = await axios.post(
            `https://${region}.api.cognitive.microsoft.com/sts/v1.0/issueToken`,
            null,
            { headers: { "Ocp-Apim-Subscription-Key": apiKey } }
        );
        res.json({ token: tokenRes.data, region });
    } catch (err) {
        console.error("Failed to generate Azure token:", err.response?.data || err.message);
        res.status(500).json({ error: "Failed to generate token" });
    }
});

// Start server
app.listen(port, () => {
    console.log(`Server running on http://localhost:${port}`);
});