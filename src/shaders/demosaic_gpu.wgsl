// SPDX-License-Identifier: GPL-3.0-or-later
// Equation port of librtprocess 9a858270acb2096e2e403d932760ee688fcac425.
// RCD: Luis Sanz Rodriguez / Ingo Weyrich (2017-2018).
// Linear LMMSE iterations=0: Gabor Horvath, Jacques Desmis, Ingo Weyrich.
// All neighbor reads use completed, separate textures. A 16-pixel even halo
// retains the scalar reference's fictitious CFA phase at reflected borders.
struct Params {
    width:u32, height:u32, pattern:u32, method:u32,
    origin_x:u32, origin_y:u32, tile_width:u32, tile_height:u32
}
struct Status {
    negative:atomic<u32>, positive:atomic<u32>, peak:atomic<u32>, invalid:atomic<u32>
}
@group(0) @binding(0) var raw_tex:texture_2d<f32>;
@group(0) @binding(1) var a_tex:texture_2d<f32>;
@group(0) @binding(2) var b_tex:texture_2d<f32>;
@group(0) @binding(3) var dst:texture_storage_2d<rgba32float,write>;
@group(0) @binding(4) var<uniform> p:Params;
@group(0) @binding(5) var<storage,read_write> status:Status;
fn mirror(x:i32,n:u32)->i32 {
    if n<=1u{
        return 0;
    }
    let period=2*(i32(n)-1);
    let t=abs(x)%period;
    return min(t,period-t);
}
fn raw(q:vec2<i32>)->f32 {
    let c=q-vec2(16);
    return textureLoad(raw_tex,vec2(mirror(c.x,p.width),mirror(c.y,p.height)),0).r;
}
fn a(q:vec2<i32>)->vec4<f32>{
    return textureLoad(a_tex,q-vec2<i32>(i32(p.origin_x),i32(p.origin_y)),0);
}
fn b(q:vec2<i32>)->vec4<f32>{
    return textureLoad(b_tex,q-vec2<i32>(i32(p.origin_x),i32(p.origin_y)),0);
}
fn channel(q:vec2<i32>)->u32 {
    let i=u32(q.y&1)*2u+u32(q.x&1);
    var s=vec4<u32>(0,1,2,3);
    if p.pattern==1u{
        s=vec4<u32>(3,2,1,0);
    }
    else if p.pattern==2u{
        s=vec4<u32>(1,0,3,2);
    }
    else if p.pattern==3u{
        s=vec4<u32>(2,3,0,1);
    }
    return select(select(1u,2u,s[i]==3u),0u,s[i]==0u);
}
fn inside(q:vec2<i32>,r:i32)->bool{
    return all(q>=vec2(r)) && all(q<vec2<i32>(i32(p.width)+32-r,i32(p.height)+32-r));
}
// Preserve authored rounding at scene scale; algebraic differences avoid
// cancellation/overflow for extreme finite constant planes.
fn hp(q:vec2<i32>,d:vec2<i32>)->f32{
    var v=(raw(q-3*d)-raw(q-d)-raw(q+d)+raw(q+3*d))-3.0*(raw(q-2*d)+raw(q+2*d))+6.0*raw(q);
    if abs(raw(q))>1e15 {
        v=(raw(q-3*d)-raw(q-d))+(raw(q+3*d)-raw(q+d))-3.0*((raw(q-2*d)-raw(q))+(raw(q+2*d)-raw(q)));
    }
    return v*v;
}
fn paired_hp(q:vec2<i32>,d:vec2<i32>)->f32{
    return hp(vec2(q.x|1,q.y),d);
}
fn weighted(x:f32,y:f32,gx:f32,gy:f32)->f32{
    if max(max(abs(x),abs(y)),max(abs(gx),abs(gy)))>1e15{
        return (gy/(gx+gy))*x+(gx/(gx+gy))*y;
    }
    return (gy*x+gx*y)/(gx+gy);
}
fn refined(x:f32,y:f32)->f32{
    return select(x,y,abs(0.5-x)<abs(0.5-y));
}
fn grad(q:vec2<i32>,d:vec2<i32>)->f32{
    return 1e-5+abs(raw(q-d)-raw(q+d))+abs(raw(q)-raw(q+2*d))+abs(raw(q+d)-raw(q+3*d))+abs(raw(q+2*d)-raw(q+4*d));
}
fn estimate(q:vec2<i32>,d:vec2<i32>)->f32{
    let l=a(q).y;
    let n=a(q+2*d).y;
    let den=1e-5+l+n;
    if l<=0.0||n<=0.0||abs(den)<1e-5{
        return raw(q+d)+0.5*(raw(q)-raw(q+2*d));
    }
    if abs(raw(q+d))>1e15{
        // Keep the ratio bounded before multiplying the scene-scale sample.
        // Scaling also keeps its reciprocal normal on GPUs that flush
        // subnormals; the clamp prevents reassociation into an overflowing
        // raw*(l+l) product. l,n are positive here, so the ratio is in [0,2].
        let sl=l*1e-20;
        let sn=n*1e-20;
        let ratio=(sl+sl)/(1e-25+sl+sn);
        return raw(q+d)*clamp(ratio,0.0,2.0);
    }
    return raw(q+d)*(l+l)/den;
}
fn cd(q:vec2<i32>,d:vec2<i32>,c:u32)->f32{
    return b(q+d)[c]-b(q+d).y;
}
fn diagonal_grad(q:vec2<i32>,d:vec2<i32>,c:u32)->f32{
    return 1e-5+abs(b(q+d)[c]-b(q-d)[c])+abs(b(q+d)[c]-b(q+3*d)[c])+abs(b(q).y-b(q+2*d).y);
}
fn axial_grad(q:vec2<i32>,d:vec2<i32>,c:u32)->f32{
    return 1e-5+abs(b(q).y-b(q+2*d).y)+abs(b(q-d)[c]-b(q+d)[c])+abs(b(q+d)[c]-b(q+3*d)[c]);
}
fn finite(v:f32)->bool{
    return (bitcast<u32>(v)&0x7f800000u)!=0x7f800000u;
}
// Tree reduction bounds global atomics to one pair per 16x16 group.
// Every lane, including the partial edge group, reaches every barrier.
var<workgroup> reduction:array<vec2<u32>,256>;
fn group_max(value:vec2<u32>,lane:u32)->vec2<u32>{
    reduction[lane]=value;
    workgroupBarrier();
    for(var stride=128u;stride>0u;stride/=2u){
        if lane<stride {
            reduction[lane]=max(reduction[lane],reduction[lane+stride]);
        }
        workgroupBarrier();
    }
    return reduction[0];
}
@compute @workgroup_size(16,16)
fn bounds(@builtin(global_invocation_id) id:vec3<u32>,@builtin(local_invocation_index) lane:u32){
    var value=vec2<u32>(0u);
    if id.x<p.width && id.y<p.height {
        let v=textureLoad(raw_tex,id.xy,0).r;
        if !finite(v){
            atomicOr(&status.invalid,1u);
        }
        else if v<0.0 {
            value.x=bitcast<u32>(-v);
        }
        else {
            value.y=bitcast<u32>(v);
        }
    }
    let global=group_max(value,lane);
    if lane==0u{
        atomicMax(&status.negative,global.x);
        atomicMax(&status.positive,global.y);
    }
}
@compute @workgroup_size(16,16)
// RCD: vertical/horizontal highpass discrimination + lowpass.
// LMMSE: horizontal/vertical green-colour differences with global bounds.
fn first(@builtin(global_invocation_id) id:vec3<u32>){
    if id.x>=p.tile_width+32u || id.y>=p.tile_height+32u {return;}
    let q=vec2<i32>(id.xy)+vec2<i32>(i32(p.origin_x),i32(p.origin_y));
    if !inside(q,0){
        return;
    }
    var v=vec4(0.0);
    if p.method==0u{
        v.x=0.5;
        if inside(q,4){
            let d=vec2(0,1);
            let e=vec2(1,0);
            let vv=max(hp(q-d,d)+hp(q,d)+hp(q+d,d),1e-10);
            let hh=max(hp(q-e,e)+hp(q,e)+hp(q+e,e),1e-10);
            v.x=vv/(vv+hh);
        }
        if inside(q,2){
            v.y=raw(q)+0.5*(raw(q-vec2(0,1))+raw(q+vec2(0,1))+raw(q-vec2(1,0))+raw(q+vec2(1,0)))+0.25*(raw(q-vec2(1,1))+raw(q+vec2(1,-1))+raw(q+vec2(-1,1))+raw(q+vec2(1,1)));
        }
    }
    else if inside(q,2){
        let lo=-bitcast<f32>(atomicLoad(&status.negative));
        let hi=bitcast<f32>(atomicLoad(&status.positive));
        let v0=0.0625*(raw(q-vec2(1,1))+raw(q+vec2(1,-1))+raw(q+vec2(-1,1))+raw(q+vec2(1,1)))+0.25*raw(q);
        for(var c=0u;c<2u;c++){
            let d=select(vec2(1,0),vec2(0,1),c==1u);
            var est=-0.25*(raw(q-2*d)+raw(q+2*d))+0.5*(raw(q-d)+raw(q)+raw(q+d));
            if abs(raw(q))>1e15 {
                est=raw(q)-0.25*((raw(q-2*d)-raw(q))+(raw(q+2*d)-raw(q)))+0.5*((raw(q-d)-raw(q))+(raw(q+d)-raw(q)));
            }
            if !finite(est) || !finite(v0) {atomicOr(&status.invalid,4u);}
            if channel(q)!=1u{
                var g=clamp(est,lo,hi);
                if raw(q)>1.75*(v0+0.5*est){
                    g=clamp(est,min(raw(q-d),raw(q+d)),max(raw(q-d),raw(q+d)));
                }
                v[c]=g-raw(q);
            }
            else{
                v[c]=clamp(-est,-hi,-lo)+raw(q);
            }
        }
    }
    for(var component=0u;component<4u;component++){
        if !finite(v[component]) {atomicOr(&status.invalid,8u);}
    }
    textureStore(dst,id.xy,v);
}
@compute @workgroup_size(16,16)
// RCD: ratio-corrected directional green estimates.
// LMMSE: separable nine-tap Gaussian smoothing of directional differences.
fn second(@builtin(global_invocation_id) id:vec3<u32>){
    if id.x>=p.tile_width+32u || id.y>=p.tile_height+32u {return;}
    let q=vec2<i32>(id.xy)+vec2<i32>(i32(p.origin_x),i32(p.origin_y));
    if !inside(q,0){
        return;
    }
    var v=vec4(0.0);
    if p.method==0u{
        v=vec4(vec3(raw(q)),0.0);
        if inside(q,5)&&channel(q)!=1u{
            let d=vec2(0,1);
            let e=vec2(1,0);
            let vv=weighted(estimate(q,-d),estimate(q,d),grad(q,-d),grad(q,d));
            let hh=weighted(estimate(q,-e),estimate(q,e),grad(q,-e),grad(q,e));
            let disc=refined(a(q).x,0.25*(a(q-vec2(1,1)).x+a(q+vec2(1,-1)).x+a(q+vec2(-1,1)).x+a(q+vec2(1,1)).x));
            v.y=disc*hh+(1.0-disc)*vv;
        }
    }
    else if inside(q,6){
        let weights=array<f32,5>(0.2041636887,0.1801738229,0.1238315368,0.0662822453,0.0276305506);
        for(var c=0u;c<2u;c++){
            let d=select(vec2(1,0),vec2(0,1),c==1u);
            v[c]=weights[0]*a(q)[c];
            for(var k=1;k<5;k++){
                v[c]+=weights[k]*(a(q-k*d)[c]+a(q+k*d)[c]);
            }
        }
    }
    for(var component=0u;component<4u;component++){
        if !finite(v[component]) {atomicOr(&status.invalid,8u);}
    }
    textureStore(dst,id.xy,v);
}
@compute @workgroup_size(16,16)
// RCD: paired-column diagonal discrimination (upstream HP phase).
// LMMSE: local signal/noise variance and minimum-MSE green fusion.
fn third(@builtin(global_invocation_id) id:vec3<u32>){
    if id.x>=p.tile_width+32u || id.y>=p.tile_height+32u {return;}
    let q=vec2<i32>(id.xy)+vec2<i32>(i32(p.origin_x),i32(p.origin_y));
    if !inside(q,0){
        return;
    }
    var v=vec4(0.0);
    if p.method==0u{
        v.x=a(q).x;
        if inside(q,5){
            let d=vec2(1,1);
            let e=vec2(-1,1);
            let pp=max(paired_hp(q-d,d)+paired_hp(q,d)+paired_hp(q+d,d),1e-10);
            let qq=max(paired_hp(q-e,e)+paired_hp(q,e)+paired_hp(q+e,e),1e-10);
            v.y=pp/(pp+qq);
        }
    }
    else if inside(q,10){
        let c=channel(q);
        v[c]=raw(q);
        if c!=1u{
            var est=vec2(0.0);
            var variance=vec2(0.0);
            for(var axis=0u;axis<2u;axis++){
                let d=select(vec2(1,0),vec2(0,1),axis==1u);
                var mu=0.0;
                for(var k=-4;k<=4;k++){
                    mu+=b(q+k*d)[axis];
                }
                mu/=9.0;
                var vx=1e-7;
                var vn=1e-7;
                for(var k=-4;k<=4;k++){
                    let j=q+k*d;
                    let x=b(j)[axis]-mu;
                    let n=b(j)[axis]-a(j)[axis];
                    vx+=x*x;
                    vn+=n*n;
                }
                est[axis]=(a(q)[axis]*vx+b(q)[axis]*vn)/(vx+vn);
                variance[axis]=vx*vn/(vx+vn);
            }
            v.y=raw(q)+weighted(est.x,est.y,variance.x,variance.y);
        }
    }
    for(var component=0u;component<4u;component++){
        if !finite(v[component]) {atomicOr(&status.invalid,8u);}
    }
    textureStore(dst,id.xy,v);
}
@compute @workgroup_size(16,16)
// RCD: opposite chroma at R/B using diagonal colour differences.
// LMMSE: missing chroma at green sites along each colour CFA axis.
fn fourth(@builtin(global_invocation_id) id:vec3<u32>){
    if id.x>=p.tile_width+32u || id.y>=p.tile_height+32u {return;}
    let q=vec2<i32>(id.xy)+vec2<i32>(i32(p.origin_x),i32(p.origin_y));
    if !inside(q,0){
        return;
    }
    var v=b(q);
    let c0=channel(q);
    if p.method==0u{
        if inside(q,7)&&c0!=1u{
            let c=2u-c0;
            let d=vec2(1,1);
            let e=vec2(-1,1);
            let pe=weighted(cd(q,-d,c),cd(q,d,c),diagonal_grad(q,-d,c),diagonal_grad(q,d,c));
            let qe=weighted(cd(q,-e,c),cd(q,e,c),diagonal_grad(q,-e,c),diagonal_grad(q,e,c));
            let disc=refined(a(q).y,0.25*(a(q-vec2(1,1)).y+a(q+vec2(1,-1)).y+a(q+vec2(-1,1)).y+a(q+vec2(1,1)).y));
            v[c]=v.y+disc*qe+(1.0-disc)*pe;
        }
    }
    else if inside(q,11)&&c0==1u{
        for(var axis=0u;axis<2u;axis++){
            let d=select(vec2(1,0),vec2(0,1),axis==1u);
            let c=channel(q+d);
            v[c]=v.y+0.5*(cd(q,-d,c)+cd(q,d,c));
        }
    }
    for(var component=0u;component<4u;component++){
        if !finite(v[component]) {atomicOr(&status.invalid,8u);}
    }
    textureStore(dst,id.xy,v);
}
@compute @workgroup_size(16,16)
// RCD: axial chroma at green. LMMSE: opposite chroma from completed green.
// Crop the even halo, retain sensor peak, and reduce output safety/colour bound.
fn final_pass(@builtin(global_invocation_id) id:vec3<u32>,@builtin(local_invocation_index) lane:u32){
    var peak=0u;
    if id.x<p.tile_width && id.y<p.tile_height {
        let output_id=id.xy+vec2(p.origin_x,p.origin_y);
        let q=vec2<i32>(output_id)+vec2(16);
        var v=b(q);
        let c0=channel(q);
        if p.method==0u&&c0==1u{
            let disc=refined(a(q).x,0.25*(a(q-vec2(1,1)).x+a(q+vec2(1,-1)).x+a(q+vec2(-1,1)).x+a(q+vec2(1,1)).x));
            for(var c=0u;c<=2u;c+=2u){
                let d=vec2(0,1);
                let e=vec2(1,0);
                let vv=weighted(cd(q,-d,c),cd(q,d,c),axial_grad(q,-d,c),axial_grad(q,d,c));
                let hh=weighted(cd(q,-e,c),cd(q,e,c),axial_grad(q,-e,c),axial_grad(q,e,c));
                v[c]=v.y+disc*hh+(1.0-disc)*vv;
            }
        }
        else if p.method==1u&&c0!=1u{
            let c=2u-c0;
            v[c]=v.y+0.25*(cd(q,vec2(0,-1),c)+cd(q,vec2(-1,0),c)+cd(q,vec2(1,0),c)+cd(q,vec2(0,1),c));
        }
        else if p.method==2u{
            let hv=0.25*(raw(q-vec2(1,0))+raw(q+vec2(1,0))+raw(q-vec2(0,1))+raw(q+vec2(0,1)));
            let diag=0.25*(raw(q-vec2(1,1))+raw(q+vec2(1,-1))+raw(q+vec2(-1,1))+raw(q+vec2(1,1)));
            if c0==0u{
                v=vec4(raw(q),hv,diag,0.0);
            }
            else if c0==2u{
                v=vec4(diag,hv,raw(q),0.0);
            }
            else{
                v.y=raw(q);
                let c=channel(q+vec2(1,0));
                v[c]=0.5*(raw(q-vec2(1,0))+raw(q+vec2(1,0)));
                v[2u-c]=0.5*(raw(q-vec2(0,1))+raw(q+vec2(0,1)));
            }
        }
        v.w=raw(q);
        for(var y=-1;y<=1;y++){
            for(var x=-1;x<=1;x++){
                if c0!=1u||x==0||y==0{
                    v.w=max(v.w,raw(q+vec2(x,y)));
                }
            }
        }
        for(var c=0u;c<4u;c++){
            if !finite(v[c]){
                atomicOr(&status.invalid,2u);
            }
            else if c<3u{
                peak=max(peak,bitcast<u32>(abs(v[c])));
            }
        }
        textureStore(dst,output_id,v);
    }
    let global=group_max(vec2(peak,0u),lane);
    if lane==0u {
        atomicMax(&status.peak,global.x);
    }
}
