// SPDX-License-Identifier: GPL-3.0-only
// Merged normalized R,Gr,B,Gb -> spatial CFA, without clamping or renormalizing.
struct Params { width:u32, height:u32, pattern:u32, method:u32 }
@group(0) @binding(0) var<storage,read> planes:array<vec4<f32>>;
@group(0) @binding(1) var dst:texture_storage_2d<r32float,write>;
@group(0) @binding(2) var<uniform> p:Params;
@compute @workgroup_size(16,16)
fn main(@builtin(global_invocation_id) id:vec3<u32>){
 if id.x>=p.width||id.y>=p.height{return;}
 var sites=vec4<u32>(0,1,3,2);
 if p.pattern==1u{sites=vec4<u32>(2,3,1,0);}else if p.pattern==2u{sites=vec4<u32>(1,0,2,3);}else if p.pattern==3u{sites=vec4<u32>(3,2,0,1);}
 let v=planes[(id.y/2u)*(p.width/2u)+id.x/2u][sites[(id.y&1u)*2u+(id.x&1u)]];
 textureStore(dst,id.xy,vec4(v,0.0,0.0,0.0));
}
