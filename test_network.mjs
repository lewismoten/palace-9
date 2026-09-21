import assert from 'node:assert/strict';
import { routeTimeline, selectedExpert, activeExperts, hitTestNetwork, signedWeightColor, strongestRoutes } from './network.mjs';

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
const routedEdges=[
 {stage:0,fromIndex:0,toIndex:0,weight:.9},{stage:0,fromIndex:0,toIndex:1,weight:.2},
 {stage:1,fromIndex:0,toIndex:0,weight:.8},{stage:1,fromIndex:1,toIndex:1,weight:1},
 {stage:2,fromIndex:0,toIndex:0,weight:.7},{stage:2,fromIndex:1,toIndex:0,weight:1},
];
const routes=strongestRoutes(routedEdges,1,[0],1);
assert.equal(routes.length,1);
assert.deepEqual(routes[0].map(edge=>edge.weight),[.9,.8,.7]);
const fourStage=[...routedEdges,{stage:3,fromIndex:0,toIndex:0,weight:.6}];
assert.deepEqual(strongestRoutes(fourStage,1,[0],1)[0].map(edge=>edge.weight),[.9,.8,.7,.6]);
console.log('network routing: ok');
