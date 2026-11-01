/* Admission is separate from ranking. A retrieval score is not a match verdict. */
(function(root){
 function status(photo){
  if(photo.fact_conflict)return 'not_matched';
  if(photo.inspection_uncertain)return 'uncertain';
  if(['matched','uncertain','unreviewed','not_matched'].includes(photo.match_status))return photo.match_status;
  if(!photo.inspection)return 'unreviewed';
  return photo.inspection_tier===2?'matched':photo.inspection_tier===0?'not_matched':'uncertain';
 }
 function partition(photos){
  const groups={matched:[],uncertain:[],unreviewed:[],not_matched:[],other:[]};
  photos.forEach(photo=>{const key=status(photo);groups[key].push(photo);if(key!=='matched')groups.other.push(photo);});
  return groups;
 }
 function candidateBatch(photos,visibleOrder,count){
  const lookup=new Map(photos.map(photo=>[photo.id,photo]));
  const ranked=photos.slice().sort((a,b)=>(a.rank_order??99999)-(b.rank_order??99999));
  return [...new Set([...visibleOrder,...ranked.map(photo=>photo.id)])].map(id=>lookup.get(id))
   .filter(photo=>photo&&status(photo)==='unreviewed').slice(0,Math.min(64,Math.max(0,Number(count)||0)));
 }
 const api={status,partition,candidateBatch};root.PhotoMatches=api;
 if(typeof module!=='undefined'&&module.exports)module.exports=api;
})(globalThis);
