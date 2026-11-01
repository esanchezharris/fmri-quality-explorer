// Labeled unit examples only. These arrays never become study observations.
import test from 'node:test';
import assert from 'node:assert/strict';
import { signalRange, linePath, nearestFrame, coverageLabel } from '../fmri_study_check/web/explorer-model.mjs';

test('constant signals receive finite plot bounds around their actual value', () => {
  const [lo, hi] = signalRange([1.94, 1.94, 1.94]);
  assert.ok(Number.isFinite(lo) && Number.isFinite(hi) && lo < 1.94 && hi > 1.94);
  assert.equal(signalRange([null, null]), null);
});

test('undefined observations create path gaps instead of zero imputation', () => {
  const path = linePath([0, 2, 4], [1, null, 3], x => x * 25, y => 30 - y * 10);
  assert.equal(path, 'M0.00,20.00M100.00,0.00');
  assert.ok(!path.includes('NaN'));
});

test('time selection retains original acquisition positions', () => {
  const times = Array.from({length:168}, (_,i) => i * 2);
  assert.equal(nearestFrame(226, times), 113);
  assert.equal(nearestFrame(999, times), 167);
  assert.equal(nearestFrame(-2, times), 0);
});

test('zero support, unknown support and empty strata have distinct labels', () => {
  assert.equal(coverageLabel({total:38,included:0,unknown:0,fraction:0}), '0.0%');
  assert.equal(coverageLabel({total:0,included:0,unknown:0,fraction:null}), 'Unavailable');
  assert.equal(coverageLabel({total:4,included:2,unknown:1,fraction:null}), 'Unknown');
});
