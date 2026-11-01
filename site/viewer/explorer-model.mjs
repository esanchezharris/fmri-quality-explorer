// Display mathematics only. Scientific observations and exclusion masks are prepared in Python.
export function signalRange(values) {
  const finite = (values || []).filter(v => v !== null && Number.isFinite(v));
  if (!finite.length) return null;
  const lo = Math.min(...finite), hi = Math.max(...finite);
  const padding = hi === lo ? Math.max(1, Math.abs(lo) * 0.03) : (hi - lo) * 0.13;
  return [lo - padding, hi + padding];
}

export function linePath(times, values, x, y) {
  let path = '', pen = false;
  values.forEach((value, i) => {
    if (value === null || !Number.isFinite(value)) { pen = false; return; }
    path += `${pen ? 'L' : 'M'}${x(times[i]).toFixed(2)},${y(value).toFixed(2)}`;
    pen = true;
  });
  return path;
}

export function nearestFrame(time, times) {
  let best = 0;
  for (let i = 1; i < times.length; i++) {
    if (Math.abs(times[i] - time) < Math.abs(times[best] - time)) best = i;
  }
  return best;
}

export function coverageLabel(partition) {
  if (!partition.total) return 'Unavailable';
  if (partition.unknown || partition.fraction === null) return 'Unknown';
  return `${(partition.fraction * 100).toFixed(1)}%`;
}
