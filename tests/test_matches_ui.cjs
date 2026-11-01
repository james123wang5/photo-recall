const assert=require('node:assert/strict');
const {partition,candidateBatch}=require('../web/matches.js');
const photos=[
 {id:'right',match_status:'matched',rank_order:0},
 {id:'unchecked',score:.999,rank_order:1},
 {id:'unclear',match_status:'uncertain',rank_order:2},
 {id:'wrong',match_status:'not_matched',rank_order:3},
 {id:'stale',match_status:'matched',inspection_uncertain:true,rank_order:4},
];
const grouped=partition(photos);
assert.deepEqual(grouped.matched.map(p=>p.id),['right']);
assert.deepEqual(grouped.other.map(p=>p.id),['unchecked','unclear','wrong','stale']);
// A filtered main view contains no unchecked cards. Continue must still reach
// the hidden candidate pool, rather than repeatedly checking accepted results.
assert.deepEqual(candidateBatch(photos,['right'],12).map(p=>p.id),['unchecked']);
assert.equal(partition(photos.filter(p=>p.id!=='right')).matched.length,0);
assert.equal(photos.length,5); // Filtering never deletes the candidate pool.
console.log('Result filtering and hidden-candidate continuation passed');
