// Run: node --test tests/
// Loads the <script id="calc-core"> block from src/calculator.html and tests it without a browser.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, '..', 'src', 'calculator.html'), 'utf8');
const src = html.match(/<script id="calc-core">([\s\S]*?)<\/script>/)[1];
const ctx = {};
vm.runInNewContext(src + '\nthis.CalcCore = CalcCore;', ctx);
const { calculate, press, addHistory, preview, pretty } = ctx.CalcCore;

// Press a sequence of keys from a blank state, e.g. type('12+3=')
function type(keys, state = { expr: '', fresh: false }) {
  const list = Array.isArray(keys) ? keys : keys.split('');
  return list.reduce(press, state);
}

test('basic arithmetic', () => {
  assert.equal(calculate('1+2'), '3');
  assert.equal(calculate('10-4'), '6');
  assert.equal(calculate('6*7'), '42');
  assert.equal(calculate('9/3'), '3');
});

test('operator precedence and parentheses', () => {
  assert.equal(calculate('2+3*4'), '14');
  assert.equal(calculate('10-6/2'), '7');
  assert.equal(calculate('(2+3)*4'), '20');
});

test('unary minus and decimals', () => {
  assert.equal(calculate('-5+2'), '-3');
  assert.equal(calculate('0.1+0.2'), '0.3');
  assert.equal(calculate('.5*4'), '2');
});

test('percent', () => {
  assert.equal(calculate('50%'), '0.5');
  assert.equal(calculate('200*10%'), '20');
});

test('errors', () => {
  assert.equal(calculate('5/0'), 'Error');
  assert.equal(calculate('2+'), 'Error');
  assert.equal(calculate('abc'), 'Error');
  assert.equal(calculate('(1+2'), 'Error');
  assert.equal(calculate(''), 'Error');
});

test('key presses: typing and equals', () => {
  const s = type('12+3=');
  assert.equal(s.expr, '15');
  assert.equal(s.fresh, true);
  assert.equal(type('2+3*4=').expr, '14');
});

test('key presses: operator replaces previous operator', () => {
  assert.equal(type('5+*').expr, '5*');
  assert.equal(type('5+*2=').expr, '10');
});

test('key presses: decimal point rules', () => {
  assert.equal(type('.').expr, '0.');
  assert.equal(type('1.2.3').expr, '1.23');
  assert.equal(type('1+.5').expr, '1+0.5');
});

test('key presses: no leading zeros', () => {
  assert.equal(type('007').expr, '7');
  assert.equal(type('5+00').expr, '5+0');
});

test('key presses: leading minus only', () => {
  assert.equal(type('*').expr, '');
  assert.equal(type('-3+1=').expr, '-2');
});

test('key presses: after result, digit starts over, operator continues', () => {
  assert.equal(type('2+2=7').expr, '7');
  assert.equal(type('2+2=*3=').expr, '12');
});

test('key presses: trailing operator ignored on equals', () => {
  assert.equal(type('8+=').expr, '8');
});

test('key presses: clear and backspace', () => {
  assert.equal(type(['1', '2', '3', 'Backspace']).expr, '12');
  assert.equal(type(['1', '+', '2', 'C']).expr, '');
  assert.equal(type(['5', '=', 'Backspace']).expr, '');
});

test('key presses: divide by zero shows Error, next key recovers', () => {
  const s = type('5/0=');
  assert.equal(s.expr, 'Error');
  assert.equal(press(s, '4').expr, '4');
});

test('key presses: keyboard aliases Enter and Escape', () => {
  assert.equal(type(['3', '*', '3', 'Enter']).expr, '9');
  assert.equal(type(['3', 'Escape']).expr, '');
});

test('history: equals produces an entry', () => {
  const s = type('12+3=');
  assert.equal(s.entry.expr, '12+3');
  assert.equal(s.entry.result, '15');
  assert.equal(type('12+3').entry, undefined);
  assert.equal(type('8+=').entry.expr, '8');
});

test('history: newest first, capped at max', () => {
  let h = [];
  h = addHistory(h, type('1+1=').entry);
  h = addHistory(h, type('2*3=').entry);
  assert.equal(h.length, 2);
  assert.equal(h[0].expr, '2*3');
  assert.equal(h[1].result, '2');

  let capped = [];
  for (let i = 1; i <= 5; i++) capped = addHistory(capped, { expr: i + '+0', result: String(i) }, 3);
  assert.equal(capped.length, 3);
  assert.equal(capped[0].result, '5');
});

test('history: skips errors, no-op entries and non-equals keys', () => {
  const h = [];
  assert.equal(addHistory(h, type('5/0=').entry), h);
  assert.equal(addHistory(h, type('8=').entry), h);
  assert.equal(addHistory(h, type('8+=').entry), h);
  assert.equal(addHistory(h, undefined), h);
});

test('preview and display formatting', () => {
  assert.equal(preview(type('2+3')), '= 5');
  assert.equal(preview(type('2+')), '= 2');
  assert.equal(preview(type('23')), '');
  assert.equal(preview(type('2+3=')), '');
  assert.equal(pretty('6*7/2-1'), '6×7÷2−1');
});
