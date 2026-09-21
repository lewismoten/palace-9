import assert from 'node:assert/strict';
import { routeTimeline, selectedExpert } from './network.mjs';

assert.equal(selectedExpert(''), 'opening');
assert.equal(selectedExpert('aebd'), 'win');
assert.deepEqual(routeTimeline('aebd').map(row => row.expert), ['opening', 'position', 'position', 'block', 'win']);
console.log('network routing: ok');
