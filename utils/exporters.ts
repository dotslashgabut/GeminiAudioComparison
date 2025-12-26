
import { TranscriptionSegment } from "../types";

/**
 * Robustly parses various timestamp formats into total seconds.
 */
const parseTimestampToSeconds = (ts: string | number): number => {
  if (ts === undefined || ts === null) return 0;
  let str = ts.toString().trim().toLowerCase();
  str = str.replace(/[ms]/g, '').replace(',', '.');

  if (str.includes(':')) {
    const parts = str.split(':').map(p => parseFloat(p) || 0);
    if (parts.length === 3) return (parts[0] * 3600) + (parts[1] * 60) + parts[2];
    if (parts.length === 2) return (parts[0] * 60) + parts[1];
  }
  return parseFloat(str) || 0;
};

const formatSecondsToSRT = (totalSeconds: number): string => {
  const roundedMs = Math.round(totalSeconds * 1000);
  const h = Math.floor(roundedMs / 3600000);
  const m = Math.floor((roundedMs % 3600000) / 60000);
  const s = Math.floor((roundedMs % 60000) / 1000);
  const ms = roundedMs % 1000;
  
  return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')},${ms.toString().padStart(3, '0')}`;
};

const formatSecondsToLRC = (totalSeconds: number): string => {
  const m = Math.floor(totalSeconds / 60);
  const s = (totalSeconds % 60).toFixed(2);
  const [ss, ms] = s.split('.');
  return `[${m.toString().padStart(2, '0')}:${ss.padStart(2, '0')}.${(ms || '00').substring(0, 2).padEnd(2, '0')}]`;
};

export const downloadFile = (content: string, filename: string) => {
  const blob = new Blob([content], { type: 'application/octet-stream' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
};

export const exportAsTXT = (segments: TranscriptionSegment[], type: 'original' | 'translated'): string => {
  return segments.map(s => {
    const text = type === 'translated' ? (s.translatedText || '') : s.text;
    return `[${s.startTime}] ${text}`;
  }).join('\n\n');
};

export const exportAsSRT = (segments: TranscriptionSegment[], type: 'original' | 'translated'): string => {
  return segments.map((s, i) => {
    const text = type === 'translated' ? (s.translatedText || '') : s.text;
    const start = formatSecondsToSRT(parseTimestampToSeconds(s.startTime));
    const end = formatSecondsToSRT(parseTimestampToSeconds(s.endTime));
    return `${i + 1}\n${start} --> ${end}\n${text}\n`;
  }).join('\n');
};

export const exportAsLRC = (segments: TranscriptionSegment[], type: 'original' | 'translated', totalDuration?: number): string => {
  const lines: string[] = [];
  for (let i = 0; i < segments.length; i++) {
    const s = segments[i];
    const startTime = parseTimestampToSeconds(s.startTime);
    const endTime = parseTimestampToSeconds(s.endTime);
    const text = type === 'translated' ? (s.translatedText || '') : s.text;

    lines.push(`${formatSecondsToLRC(startTime)}${text}`);

    const nextS = segments[i + 1];
    if (nextS) {
      const nextStartTime = parseTimestampToSeconds(nextS.startTime);
      const clearingTime = endTime + 2;
      if (nextStartTime > clearingTime) {
        lines.push(`${formatSecondsToLRC(clearingTime)}`);
      }
    } else {
      const clearingTime = endTime + 2;
      if (totalDuration !== undefined) {
        if (clearingTime <= totalDuration) {
          lines.push(`${formatSecondsToLRC(clearingTime)}`);
        }
      } else {
        lines.push(`${formatSecondsToLRC(clearingTime)}`);
      }
    }
  }
  return lines.join('\n');
};

export const exportAsJSON = (segments: TranscriptionSegment[], type: 'original' | 'translated'): string => {
  const data = segments.map(s => ({
    startTime: s.startTime,
    endTime: s.endTime,
    text: type === 'translated' ? (s.translatedText || '') : s.text
  }));
  return JSON.stringify(data, null, 2);
};
