"use strict";

const sttStatus = document.getElementById("stt-status");
const sttOutput = document.getElementById("stt-output");
const sttMeta = document.getElementById("stt-meta");
const ttsStatus = document.getElementById("tts-status");
const ttsAudio = document.getElementById("tts-audio");
const ttsDownload = document.getElementById("tts-download");

let mediaRecorder = null;
let recordedChunks = [];

document.getElementById("stt-record").addEventListener("click", async () => {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  recordedChunks = [];
  mediaRecorder = new MediaRecorder(stream);
  mediaRecorder.ondataavailable = (e) => { if (e.data.size) recordedChunks.push(e.data); };
  mediaRecorder.onstop = () => stream.getTracks().forEach(t => t.stop());
  mediaRecorder.start();
  sttStatus.textContent = "recording…";
  document.getElementById("stt-stop").disabled = false;
  document.getElementById("stt-record").disabled = true;
});

document.getElementById("stt-stop").addEventListener("click", async () => {
  mediaRecorder.stop();
  document.getElementById("stt-stop").disabled = true;
  document.getElementById("stt-record").disabled = false;
  sttStatus.textContent = "transcribing…";
  await new Promise(r => setTimeout(r, 100));
  const blob = new Blob(recordedChunks, { type: "audio/webm" });
  await transcribeBlob(blob);
});

document.getElementById("stt-file").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  sttStatus.textContent = "transcribing…";
  await transcribeBlob(file);
});

async function transcribeBlob(blob) {
  const fd = new FormData();
  fd.append("audio", blob, "recording.webm");
  const lang = document.getElementById("stt-language").value;
  if (lang) fd.append("language", lang);
  fd.append("vad", document.getElementById("stt-vad").checked ? "true" : "false");
  const t0 = performance.now();
  const r = await fetch("/v1/transcribe", { method: "POST", body: fd });
  if (!r.ok) {
    sttStatus.textContent = "error: " + r.status;
    return;
  }
  const body = await r.json();
  const elapsed = Math.round(performance.now() - t0);
  sttOutput.value = body.text;
  sttMeta.textContent = `lang: ${body.language} (p=${body.language_probability.toFixed(2)}) · duration: ${body.duration.toFixed(2)}s · client rtt: ${elapsed}ms`;
  sttStatus.textContent = "done";
}

document.querySelectorAll('input[name="tts-mode"]').forEach(el => {
  el.addEventListener("change", () => {
    document.getElementById("tts-custom-voice").classList.toggle("hidden",
      el.value !== "custom_voice" || !el.checked);
    document.getElementById("tts-voice-design").classList.toggle("hidden",
      el.value !== "voice_design" || !el.checked);
  });
});

document.getElementById("cv-speak").addEventListener("click", async () => {
  await speak({
    mode: "custom_voice",
    text: document.getElementById("cv-text").value,
    language: document.getElementById("cv-language").value,
    speaker: document.getElementById("cv-speaker").value,
    instruct: document.getElementById("cv-instruct").value || null,
  });
});

document.getElementById("vd-speak").addEventListener("click", async () => {
  await speak({
    mode: "voice_design",
    text: document.getElementById("vd-text").value,
    language: document.getElementById("vd-language").value,
    voice_prompt: document.getElementById("vd-voice-prompt").value,
    instruct: document.getElementById("vd-instruct").value || null,
  });
});

async function speak(body) {
  ttsStatus.textContent = "synthesising…";
  ttsDownload.classList.add("hidden");
  const r = await fetch("/v1/speak", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    ttsStatus.textContent = "error: " + r.status;
    return;
  }
  const blob = await r.blob();
  const url = URL.createObjectURL(blob);
  ttsAudio.src = url;
  ttsAudio.play();
  ttsDownload.href = url;
  ttsDownload.classList.remove("hidden");
  ttsStatus.textContent = "playing";
}

document.getElementById("rt-run").addEventListener("click", async () => {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  recordedChunks = [];
  mediaRecorder = new MediaRecorder(stream);
  mediaRecorder.ondataavailable = (e) => { if (e.data.size) recordedChunks.push(e.data); };
  document.getElementById("rt-status").textContent = "recording 3s…";
  mediaRecorder.start();
  await new Promise(r => setTimeout(r, 3000));
  mediaRecorder.stop();
  stream.getTracks().forEach(t => t.stop());
  await new Promise(r => setTimeout(r, 200));
  const blob = new Blob(recordedChunks, { type: "audio/webm" });
  document.getElementById("rt-status").textContent = "transcribing…";
  await transcribeBlob(blob);
  const mode = document.querySelector('input[name="tts-mode"]:checked').value;
  if (mode === "custom_voice") {
    document.getElementById("cv-text").value = sttOutput.value;
    document.getElementById("rt-status").textContent = "speaking back…";
    document.getElementById("cv-speak").click();
  } else {
    document.getElementById("vd-text").value = sttOutput.value;
    document.getElementById("rt-status").textContent = "speaking back…";
    document.getElementById("vd-speak").click();
  }
});
