
import { GoogleGenAI, Type } from "@google/genai";
import { TranscriptionSegment } from "../types";

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY || '' });

const TRANSCRIPTION_SCHEMA = {
  type: Type.OBJECT,
  properties: {
    segments: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          startTime: {
            type: Type.STRING,
            description: "Start timestamp. MUST use HH:MM:SS.mmm format (e.g. '00:00:01.234').",
          },
          endTime: {
            type: Type.STRING,
            description: "End timestamp. MUST use HH:MM:SS.mmm format (e.g. '00:00:04.567').",
          },
          text: {
            type: Type.STRING,
            description: "Transcribed text.",
          },
        },
        required: ["startTime", "endTime", "text"],
      },
    },
  },
  required: ["segments"],
};

/**
 * Robustly normalizes timestamp strings to HH:MM:SS.mmm
 */
function normalizeTimestamp(ts: string): string {
  if (!ts) return "00:00:00.000";
  
  const raw = ts.trim();

  // Handle raw seconds format (e.g., "123.456")
  if (/^\d+(\.\d+)?$/.test(raw) && !raw.includes(':')) {
    const totalSeconds = parseFloat(raw);
    const h = Math.floor(totalSeconds / 3600);
    const m = Math.floor((totalSeconds % 3600) / 60);
    const s = Math.floor(totalSeconds % 60);
    const ms = Math.round((totalSeconds % 1) * 1000);
    
    return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${String(ms).padStart(3, '0')}`;
  }
  
  const clean = raw.replace(/[^\d:.]/g, '');
  const components = clean.split(/[:.]/);
  
  let hh = "00", mm = "00", ss = "00", mmm = "000";

  if (components.length >= 4) {
    [hh, mm, ss, mmm] = components;
  } else if (components.length === 3) {
    if (components[2].length === 3) {
      [mm, ss, mmm] = components;
    } else {
      [hh, mm, ss] = components;
    }
  } else if (components.length === 2) {
    [mm, ss] = components;
  } else if (components.length === 1) {
    ss = components[0];
  }

  return `${hh.padStart(2, '0').substring(0, 2)}:${mm.padStart(2, '0').substring(0, 2)}:${ss.padStart(2, '0').substring(0, 2)}.${mmm.padEnd(3, '0').substring(0, 3)}`;
}

/**
 * Attempts to repair truncated JSON strings.
 * ENHANCED: Now handles single-quoted JSON values or unescaped quotes more robustly
 * to prevent skipping segments when the model outputs "bad" JSON.
 */
function tryRepairJson(jsonString: string): any {
  const trimmed = jsonString.trim();

  // 1. Try standard parsing first
  try {
    const parsed = JSON.parse(trimmed);
    if (parsed.segments && Array.isArray(parsed.segments)) {
      return parsed;
    }
  } catch (e) {
    // Continue to repair logic
  }

  // 2. Try to close truncated JSON
  const lastObjectEnd = trimmed.lastIndexOf('}');
  if (lastObjectEnd !== -1) {
    const repaired = trimmed.substring(0, lastObjectEnd + 1) + "]}";
    try {
      const parsed = JSON.parse(repaired);
      if (parsed.segments && Array.isArray(parsed.segments)) {
        return parsed;
      }
    } catch (e) {
      // Continue to regex fallback
    }
  }

  // 3. Regex Fallback (The "Scraper" approach)
  // This is crucial for handling cases where quotes are messed up.
  // It looks for the pattern regardless of surrounding JSON validity.
  // Supports both "text": "value" AND "text": 'value' (in case model goes rogue)
  const segments = [];
  
  // Regex Explanation:
  // Part 1: startTime key and value (double quotes)
  // Part 2: endTime key and value (double quotes)
  // Part 3: text key... AND THEN:
  //         EITHER match double-quoted string: "((?:[^"\\]|\\.)*)"
  //         OR match single-quoted string: '((?:[^'\\]|\\.)*)'
  const segmentRegex = /\{\s*"startTime"\s*:\s*"([^"]+)"\s*,\s*"endTime"\s*:\s*"([^"]+)"\s*,\s*"text"\s*:\s*(?:"((?:[^"\\]|\\.)*)"|'((?:[^'\\]|\\.)*)')/g;
  
  let match;
  while ((match = segmentRegex.exec(trimmed)) !== null) {
    // match[1] = startTime
    // match[2] = endTime
    // match[3] = text (if double quoted)
    // match[4] = text (if single quoted)
    const rawText = match[3] !== undefined ? match[3] : match[4];
    
    // We need to unescape manually since we bypassed JSON.parse
    let unescapedText = rawText;
    try {
      // A quick hack to unescape standard JSON escapes using JSON.parse on a string fragment
      unescapedText = JSON.parse(`"${rawText.replace(/"/g, '\\"')}"`); 
    } catch (e) {
      // If that fails, just basic replace
      unescapedText = rawText.replace(/\\"/g, '"').replace(/\\'/g, "'").replace(/\\\\/g, "\\");
    }

    segments.push({
      startTime: match[1],
      endTime: match[2],
      text: unescapedText
    });
  }
  
  if (segments.length > 0) {
    return { segments };
  }
  
  throw new Error("Response structure invalid and could not be repaired.");
}

export async function transcribeAudio(
  modelName: string,
  audioBase64: string,
  mimeType: string,
  signal?: AbortSignal
): Promise<TranscriptionSegment[]> {
  try {
    const isGemini3 = modelName.includes('gemini-3');
    
    const timingPolicy = `
    STRICT TIMING POLICY:
    1. ANTI-DRIFT: Do NOT predict timestamps based on patterns or rhythm. Use ACTUAL vocal onset for 'startTime'.
    2. REPETITION HANDLING: If multiple lines start with the same text, DO NOT advance the next timestamp prematurely. Each repetition must wait for the actual audio cue.
    3. TEMPORAL ISOLATION: The 'endTime' of a segment must be precisely when the vocal stops.
    4. NO HALLUCINATION: If there is a silence between repetitions, the timestamps must reflect that silence.
    `;

    const verbatimPolicy = `
    VERBATIM & SEGMENTATION:
    1. GRANULAR SEGMENTS: Create segments shorter than 5 seconds.
    2. FULL VERBATIM: Transcribe every single word, including "uhs", "umms", and repeated phrases.
    3. NO DEDUPLICATION: If a phrase repeats 3 times, produce 3 distinct segments.
    `;

    // Added explicit instruction for JSON safety regarding quotes
    const jsonSafetyPolicy = `
    JSON FORMATTING SAFETY:
    1. TEXT ESCAPING: The 'text' field MUST be wrapped in DOUBLE QUOTES (").
    2. INTERNAL QUOTES: If the text contains a double quote, ESCAPE IT (e.g. \\"). 
    3. SINGLE QUOTES: Single quotes (') in the text are allowed but MUST NOT break the JSON structure.
    4. NO SINGLE QUOTE KEYS: Do NOT use single quotes for JSON keys (e.g. use "startTime", NOT 'startTime').
    `;

    const requestConfig: any = {
      responseMimeType: "application/json",
      responseSchema: TRANSCRIPTION_SCHEMA,
      temperature: 0, 
    };

    if (isGemini3) {
      requestConfig.thinkingConfig = { thinkingBudget: 4096 };
    }

    const abortPromise = new Promise<never>((_, reject) => {
      if (signal?.aborted) reject(new DOMException("Aborted", "AbortError"));
      signal?.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")));
    });

    const response: any = await Promise.race([
      ai.models.generateContent({
        model: modelName,
        contents: [
          {
            parts: [
              {
                inlineData: {
                  data: audioBase64,
                  mimeType: mimeType,
                },
              },
              {
                text: `Perform high-precision audio transcription with millisecond-accurate timestamps.
                
                ${timingPolicy}
                ${verbatimPolicy}
                ${jsonSafetyPolicy}
                
                REQUIRED FORMAT: JSON object with "segments" array. 
                Use HH:MM:SS.mmm for all timestamps.`,
              },
            ],
          },
        ],
        config: requestConfig,
      }),
      abortPromise
    ]);

    let text = response.text;
    if (!text) throw new Error("Empty response from model");

    text = text.trim();
    if (text.startsWith('```json')) {
      text = text.replace(/^```json\s*/, '').replace(/\s*```$/, '');
    } else if (text.startsWith('```')) {
      text = text.replace(/^```\s*/, '').replace(/\s*```$/, '');
    }

    const parsed = tryRepairJson(text);
    const segments = parsed.segments || [];

    return segments.map((s: any) => ({
      startTime: normalizeTimestamp(String(s.startTime)),
      endTime: normalizeTimestamp(String(s.endTime)),
      text: String(s.text)
    }));
  } catch (error: any) {
    if (error.name === 'AbortError') throw error;
    console.error(`Error with ${modelName}:`, error);
    throw new Error(error.message || "Transcription failed");
  }
}

export async function translateSegments(
  segments: TranscriptionSegment[],
  targetLanguage: string
): Promise<TranscriptionSegment[]> {
  try {
    const response = await ai.models.generateContent({
      model: "gemini-3-flash-preview",
      contents: [
        {
          parts: [
            {
              text: `Translate the following segments into ${targetLanguage}. 
              CRITICAL: Do NOT modify the timestamps. Keep the exact HH:MM:SS.mmm format.
              Data: ${JSON.stringify(segments)}`,
            },
          ],
        },
      ],
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            segments: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  startTime: { type: Type.STRING },
                  endTime: { type: Type.STRING },
                  text: { type: Type.STRING },
                  translatedText: { type: Type.STRING },
                },
                required: ["startTime", "endTime", "text", "translatedText"],
              },
            },
          },
        },
      },
    });

    let text = response.text;
    if (!text) throw new Error("Empty translation response");
    
    text = text.trim();
    if (text.startsWith('```json')) {
      text = text.replace(/^```json\s*/, '').replace(/\s*```$/, '');
    } else if (text.startsWith('```')) {
      text = text.replace(/^```\s*/, '').replace(/\s*```$/, '');
    }

    const parsed = JSON.parse(text);
    return parsed.segments || [];
  } catch (error: any) {
    console.error("Translation error:", error);
    throw error;
  }
}

export async function generateSpeech(text: string): Promise<string | undefined> {
  try {
    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash-preview-tts",
      contents: [{ parts: [{ text: text }] }],
      config: {
        responseModalities: ["AUDIO"],
        speechConfig: {
          voiceConfig: {
            prebuiltVoiceConfig: { voiceName: 'Zephyr' },
          },
        },
      },
    });

    return response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
  } catch (error) {
    console.error("TTS error:", error);
    throw error;
  }
}
