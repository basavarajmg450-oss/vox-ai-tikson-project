/* Unit tests for the pure end-of-speech (silence) detection logic in
 * static/js/vad.js.  No browser, network or dependencies required:
 *
 *     node tests/js/test_vad.mjs
 *
 * Covered: RMS framing, threshold behaviour, speech-then-silence auto-stop
 * timing, brief pauses that must NOT trigger, rooms that never contain
 * speech, the minimum-speech/minimum-duration guards, the hard recording
 * cap and the stalled-frame-gap clamp.
 */
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const VoxVAD = require('../../static/js/vad.js');

const { createSilenceDetector, frameRms, DEFAULTS } = VoxVAD;

// --------------------------------------------------------------- helpers
// Feed frames spaced `dtMs` apart starting at t0; returns list of verdicts.
function feed(detector, t0, frames) {
  const verdicts = [];
  let t = t0;
  for (const [rms, dtMs] of frames) {
    verdicts.push(detector.push(rms, t));
    t += dtMs;
  }
  return { verdicts, end: t };
}

const SILENCE = 0.0;
const LOUD = 0.25; // comfortably above the 0.02 default threshold

// ---------------------------------------------------------------- frameRms
{
  // Digital silence (all mid-scale bytes) has zero energy.
  assert.equal(frameRms(new Uint8Array(1024).fill(128)), 0);
  // A loud square wave has RMS ~ amplitude (alternating -127/+127 around 128).
  const square = new Uint8Array(1024);
  for (let i = 0; i < square.length; i++) square[i] = i % 2 ? 255 : 1;
  assert.ok(Math.abs(frameRms(square) - 0.992) < 0.01, 'square wave RMS ~1');
  // Constant offset near silence reads as near-zero.
  assert.ok(frameRms(new Uint8Array(64).fill(130)) < 0.02);
  // Degenerate inputs never throw.
  assert.equal(frameRms(null), 0);
  assert.equal(frameRms(new Uint8Array(0)), 0);
  console.log('ok  frameRms');
}

// ------------------------------------------- a quiet room never auto-stops
{
  const d = createSilenceDetector();
  const frames = [];
  for (let i = 0; i < 400; i++) frames.push([SILENCE, 60]); // 24 s of silence
  const { verdicts } = feed(d, 0, frames);
  assert.equal(verdicts.includes('end_of_speech'), false, 'pure silence must not trigger');
  assert.equal(d.state().speechSeen, false);
  console.log('ok  quiet room never auto-stops');
}

// ---------------------------------------- speech then 1.5 s silence triggers
{
  const d = createSilenceDetector();
  const frames = [];
  for (let i = 0; i < 50; i++) frames.push([LOUD, 60]);      // 3 s of speech
  for (let i = 0; i < 30; i++) frames.push([SILENCE, 60]);   // 1.8 s silence
  const { verdicts } = feed(d, 0, frames);
  const idx = verdicts.indexOf('end_of_speech');
  assert.notEqual(idx, -1, 'must auto-stop after the quiet window');
  // Silence had been running ~1.5 s at that point (25 quiet frames at 60 ms),
  // so the trigger lands within +-3 frames of the 25th silent frame.
  assert.ok(idx >= 50 + 22 && idx <= 50 + 28, `triggered at frame ${idx}, expected ~75`);
  console.log('ok  speech then silence auto-stops at ~1.5 s');
}

// --------------------------------- brief pauses (< silence window) do NOT fire
{
  const d = createSilenceDetector({ silenceMs: 1500 });
  const frames = [];
  for (let i = 0; i < 17; i++) frames.push([LOUD, 60]);    // speech
  for (let i = 0; i < 12; i++) frames.push([SILENCE, 60]); // 720 ms pause
  for (let i = 0; i < 17; i++) frames.push([LOUD, 60]);    // keeps talking
  for (let i = 0; i < 12; i++) frames.push([SILENCE, 60]); // 720 ms pause
  const { verdicts } = feed(d, 0, frames);
  assert.equal(verdicts.includes('end_of_speech'), false, 'mid-utterance pause must not fire');
  console.log('ok  brief pauses do not trigger');
}

// ------------------------- speech that is too short leaves silence ignored
{
  const d = createSilenceDetector(); // minSpeechMs = 250 (~4 frames at 60 ms)
  const frames = [[LOUD, 60], [LOUD, 60]]; // 120 ms of speech only
  for (let i = 0; i < 100; i++) frames.push([SILENCE, 60]); // 6 s silence
  const { verdicts } = feed(d, 0, frames);
  assert.equal(verdicts.includes('end_of_speech'), false, 'speech below minSpeechMs never auto-stops');
  console.log('ok  too-short speech leaves recording open');
}

// --------------------------------------------- hard cap at maxRecordingMs
{
  const d = createSilenceDetector({ maxRecordingMs: 3000 });
  const frames = [];
  for (let i = 0; i < 100; i++) frames.push([LOUD, 60]); // speech forever
  const { verdicts } = feed(d, 0, frames);
  const idx = verdicts.indexOf('max_duration');
  assert.notEqual(idx, -1, 'cap verdict must appear');
  assert.ok(idx >= 48 && idx <= 52, `cap at frame ${idx} (~3 s), expected ~50`);
  console.log('ok  hard recording cap');
}

// ---------------------------------------- stalled-tab gap is clamped (1 s)
{
  const d = createSilenceDetector();
  const a = d.push(LOUD, 0);
  const b = d.push(LOUD, 60);
  assert.equal(a, 'speaking');
  assert.equal(b, 'speaking');
  // Tab stalled for 10 s, then one silent frame: clamped to maxFrameGapMs
  // (1 s) of silence, so nothing triggers yet.
  assert.equal(d.push(SILENCE, 10060), 'silence');
  assert.equal(d.state().silenceMs, 1000, 'gap clamped to 1 s');
  console.log('ok  stalled frame gap is clamped');
}

// ------------------------------------------------------- custom thresholds
{
  const d = createSilenceDetector({ silenceMs: 600, speechThreshold: 0.5 });
  const frames = [[0.4, 60], [0.4, 60], [0.4, 60]]; // below custom threshold
  for (let i = 0; i < 20; i++) frames.push([SILENCE, 60]);
  const { verdicts } = feed(d, 0, frames);
  assert.equal(verdicts.includes('end_of_speech'), false, '0.4 < custom 0.5 threshold counts as silence');
  const d2 = createSilenceDetector({ silenceMs: 600, speechThreshold: 0.5 });
  const f2 = [];
  for (let i = 0; i < 10; i++) f2.push([0.6, 60]);
  for (let i = 0; i < 20; i++) f2.push([SILENCE, 60]);
  const r2 = feed(d2, 0, f2);
  assert.notEqual(r2.verdicts.indexOf('end_of_speech'), -1, '0.6 passes the custom threshold');
  console.log('ok  custom thresholds honoured');
}

// ------------------------------------------------------------ sanity: defaults
assert.ok(DEFAULTS.silenceMs >= 1000 && DEFAULTS.silenceMs <= 2000,
  'default silence window must be 1-2 s per the requirement');

console.log('All VAD logic tests passed.');
