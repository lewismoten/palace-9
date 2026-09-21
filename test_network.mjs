import assert from 'node:assert/strict';
import { routeTimeline, selectedExpert, activeExperts, hitTestNetwork, signedWeightColor } from './network.mjs';

assert.equal(selectedExpert(''), 'opening');
assert.equal(selectedExpert('aebd'), 'win');
assert.deepEqual(routeTimeline('aebd').map(row => row.expert), ['opening', 'position', 'position', 'block', 'win']);
assert.deepEqual(activeExperts('aebd'),['win','position']);
const graph={nodes:[{kind:'node',layer:'layer 1',index:3,x:10,y:10,bias:.125}],edges:[{kind:'edge',from:'input a',to:'layer 1 unit 1',x1:20,y1:20,x2:80,y2:20,weight:-.5}]};
assert.match(hitTestNetwork(graph,10,10).text,/bias \+0\.1250/);
assert.match(hitTestNetwork(graph,50,22).text,/weight −0\.5000/);
assert.equal(signedWeightColor(.5,.7),'rgba(246,166,92,0.7)');
assert.equal(signedWeightColor(-.5,.7),'rgba(121,203,229,0.7)');
assert.equal(signedWeightColor(0,.7),'rgba(166,175,164,0.7)');
console.log('network routing: ok');
