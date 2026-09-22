import assert from 'node:assert/strict';
import { clickSquare, modelMove, shouldAutoMove, autoplayDelay } from './game-controller.mjs';
import { optimalPolicy } from './rules.mjs';

const first=clickSquare({history:'',finalMove:'',seed:1983},'a');
assert.equal(first.history,'a','a board click records the first raw-history move');
assert.equal(first.finalMove,'','the first eight board placements stay in causal history');

const final=clickSquare({history:'abcdefhg',finalMove:'',seed:1983},'i');
assert.equal(final.history,'abcdefhg','the ninth placement stays out of the eight-token causal history');
assert.equal(final.finalMove,'i');

assert.equal(shouldAutoMove(1,{history:'a',finalMove:''}),true,'one-player mode answers immediately after the human X move');
assert.equal(shouldAutoMove(1,{history:'',finalMove:''}),false,'one-player mode lets the human open as X');
assert.equal(shouldAutoMove(2,{history:'a',finalMove:''}),false,'two-player mode never fills the inspected move');
assert.equal(shouldAutoMove(0,{history:'',finalMove:''}),true,'zero-player mode lets the policy open');

const response=modelMove(first);
assert.ok(optimalPolicy('a').optimal.includes(response.history.at(-1)),'model response remains in the deterministic oracle optimal set');
assert.equal(response.seed,1984,'each policy turn advances the varied-best-choice seed');

assert.equal(autoplayDelay(0),900,'first policy-vs-policy round begins at a readable pace');
assert.ok(autoplayDelay(5)<autoplayDelay(1),'policy self-play accelerates after completed games');
assert.equal(autoplayDelay(4),20,'the fifth policy-vs-policy game reaches the intentionally hard-to-follow maximum speed');
assert.equal(autoplayDelay(99),20,'policy self-play stays at its maximum speed after game five');

console.log('game controller modes: ok');
