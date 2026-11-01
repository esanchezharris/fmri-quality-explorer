import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';

test('initial payload is delivered even when child-ready happened before parent boot',()=>{
  const source=readFileSync(new URL('../site/site.js',import.meta.url),'utf8');
  const bridge=source.match(/function deliver\(\) \{([\s\S]*?)\n\}/)[0];
  const messages=[];
  const payload={participant_id:'identity-only bridge fixture'};
  const context={payload,viewerReady:false,location:{origin:'https://example.invalid'},
    $:()=>({contentWindow:{postMessage:(message,origin)=>messages.push({message,origin})}})};
  vm.runInNewContext(`${bridge}\ndeliver();`,context);
  assert.equal(messages.length,1);
  assert.equal(messages[0].message.args.data,payload);
  assert.equal(messages[0].origin,context.location.origin);
  context.payload=null;
  vm.runInNewContext(`${bridge}\ndeliver();`,context);
  assert.equal(messages.length,1,'no empty payload replaces a saved participant');
});
