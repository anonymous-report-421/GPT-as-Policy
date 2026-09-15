// Offline regression checks for the two approved build-dependency repairs.
// Run after npm ci in app/: node --test hybrid_rollout/report_site/security_smoke.test.mjs
import assert from 'node:assert/strict';
import {mkdtempSync, writeFileSync, rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createRequire} from 'node:module';
import test from 'node:test';

const require=createRequire(new URL('./app/package.json',import.meta.url));
const {customAlphabet,customRandom}=require('nanoid');
const PreviousMap=require('./node_modules/postcss/lib/previous-map.js');

test('Nano ID returns immediately for zero-size custom generators',()=>{
  assert.equal(customAlphabet('abc',0)(),'');
  assert.equal(customAlphabet('abc',4)(0),'');
  assert.equal(customRandom('abc',0,()=>{throw new Error('Randomness should not be read');})(),'');
});

test('PostCSS does not load an untrusted source map when from is absent',()=>{
  const directory=mkdtempSync(join(tmpdir(),'report-postcss-security-'));
  try{
    const path=join(directory,'fixture.map');
    const marker='SYNTHETIC_MAP_CONTENT_FOR_SECURITY_TEST';
    const content=JSON.stringify({version:3,file:'fixture.css',sources:['fixture.js'],sourcesContent:[marker],names:[],mappings:'AAAA'});
    writeFileSync(path,content);
    const css=`a { color: black }\n/*# sourceMappingURL=${path} */`;
    const untrusted=new PreviousMap(css,{});
    assert.equal(untrusted.text,undefined);
    assert.equal(untrusted.mapFile,undefined);
    // Positive control: the same valid test file is readable through an explicit
    // trusted map callback, not through the untrusted CSS annotation.
    const trusted=new PreviousMap(css,{map:{prev:()=>path}});
    assert(trusted.text.includes(marker));
  }finally{
    rmSync(directory,{recursive:true,force:true});
  }
});
