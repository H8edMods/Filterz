import {FaceLandmarker,FilesetResolver} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs';
const $=s=>document.querySelector(s),vid=$('#v'),out=$('#o'),ctx=out.getContext('2d'),glc=document.createElement('canvas'),g=glc.getContext('webgl',{preserveDrawingBuffer:true}),rnd=Math.random;
// ---------- WebGL: color grading, distortions and face warps ----------
const FS=`precision highp float;varying vec2 v;uniform sampler2D T;uniform vec2 R;uniform float t,m,mir;uniform vec4 W[4],tint;
#define K(a) (abs(m-float(a))<.5)
vec3 S(vec2 p){p=clamp(p,0.,1.);if(mir>.5)p.x=1.-p.x;return texture2D(T,p).rgb;}
float L(vec3 c){return dot(c,vec3(.299,.587,.114));}
float h(vec2 p){return fract(sin(dot(p,vec2(12.9898,78.233)))*43758.5453);}
float E(vec2 p){vec2 q=1./R;return abs(L(S(p+vec2(q.x,0.)))-L(S(p-vec2(q.x,0.))))+abs(L(S(p+vec2(0.,q.y)))-L(S(p-vec2(0.,q.y))));}
void main(){vec2 p=v,a=vec2(R.x/R.y,1.);
for(int i=0;i<4;i++){vec4 w=W[i];if(w.z>0.){vec2 d=(p-w.xy)*a;float r=length(d)/w.z;if(r<1.)p=w.xy+(p-w.xy)*(1.-w.w*pow(1.-r*r,2.));}}
vec2 c=p-.5;float r=length(c*a);
if(K(27)){float an=(1.-smoothstep(0.,.7,r))*3.*sin(t);c=mat2(cos(an),-sin(an),sin(an),cos(an))*c;p=c+.5;}
if(K(28))p=.5+c*(.45+r);
if(K(30))p=.5+c*(.85+.15*sin(t*4.));
if(K(31))p=.5+c*max(.3,1.6-1.7*r);
if(K(32))p.x+=sin(p.y*20.+t*3.)*.015;
if(K(23))p+=vec2(sin(p.y*25.+t*2.),cos(p.x*25.+t*2.))*.006;
if(K(24))p+=vec2(sin(t*2.+p.y*6.),cos(t*1.7+p.x*6.))*.01;
if(K(25)&&p.x>.5)p.x=1.-p.x;
if(K(26)){float an=atan(c.y,c.x),k=1.0472;an=abs(mod(an,k)-k*.5);p=.5+vec2(cos(an),sin(an))*length(c);}
vec3 col=S(p);float g=L(col);vec2 q=1./R;
if(K(1))col=vec3(g);
if(K(2))col=vec3(g)*vec3(1.2,1.,.8);
if(K(3)){col=mix(col,vec3(g),.3)*vec3(1.1,1.,.85)+(h(p*R+t)-.5)*.12;col*=1.-dot(c,c)*1.4;}
if(K(4)){col=vec3(smoothstep(.15,.85,g))*(1.-dot(c,c)*1.5);}
if(K(5))col*=vec3(.8,.95,1.2);
if(K(6))col*=vec3(1.2,1.,.8);
if(K(7)){float e=E(p);vec3 nc=.5+.5*cos(t+vec3(0.,2.,4.)+p.x*6.);col=col*.2+nc*e*8.;}
if(K(8))col=vec3(smoothstep(.3,.7,g),smoothstep(0.,.5,g)*smoothstep(1.,.5,g)*2.,smoothstep(.5,0.,g)+smoothstep(.8,1.,g));
if(K(9)){col=vec3(.1,1.,.2)*g*1.4+(h(p*R+t)-.5)*.2;col*=1.-dot(c,c)*1.6;}
if(K(10)){float b=floor(p.y*12.+floor(t*8.)),on=step(.8,h(vec2(b,floor(t*8.)))),o=on*(h(vec2(b,1.))-.5)*.15;col=vec3(S(p+vec2(o+.01*on,0.)).r,S(p+vec2(o,0.)).g,S(p+vec2(o-.01*on,0.)).b);}
if(K(11)){col=vec3(S(p+vec2(.006,0.)).r,g,S(p-vec2(.006,0.)).b);col=mix(col,vec3(S(p).g),.0);col*=.85+.15*sin(p.y*R.y*1.5);col+=(h(p*R+t)-.5)*.1;}
if(K(12))col=S(floor(p*R/10.)*10./R+q*5.);
if(K(13)){col=floor(col*4.+.5)/4.;if(E(p)>.12)col*=.15;}
if(K(14))col=vec3(.95,.93,.88)*(1.-clamp(E(p)*7.,0.,1.));
if(K(15)){vec2 cl=fract(p*R/8.)-.5;col=mix(vec3(1.),col*1.2,step(length(cl),sqrt(1.-g)*.75));}
if(K(16)||K(33)){vec3 b=vec3(0.);for(int i=0;i<8;i++){float an=float(i)*.785;b+=S(p+vec2(cos(an),sin(an))*.012);}b/=8.;col=K(16)?mix(col,b,.5)+.05:col+max(b-.5,0.)*1.5;}
if(K(17))col=mix(vec3(.1,0.,.3),vec3(1.,.4,.7),g);
if(K(18)){col=vec3(S(p+vec2(.004,0.)).r,g*.8,S(p-vec2(.004,0.)).b)*vec3(1.1,.8,1.3)+vec3(0.,.05,.15);col=(col-.5)*1.3+.5;}
if(K(19)){float ch=h(vec2(floor(p.x*R.x/12.),0.));float y=fract(p.y+t*.5*(.3+ch)+ch);col=vec3(0.,g*1.2,0.)+vec3(0.,1.,.3)*pow(y,6.)*.6;}
if(K(20))col=1.-col;
if(K(21))col=vec3(1.-g)*vec3(.7,.9,1.2);
if(K(22))col=vec3(S(p+vec2(.012,0.)).r,g*0.+S(p).g,S(p-vec2(.012,0.)).b);
if(K(23))col*=vec3(.6,.9,1.2);
if(K(24))col=(col+S(p+vec2(.02*sin(t*3.),0.)))*.5;
if(K(34))col=floor(col*3.)/3.*vec3(1.2,1.,1.3);
if(K(35))col=col*vec3(1.2,1.05,.8)+vec3(.05,.02,0.);
col=mix(col,col*tint.rgb,tint.a);gl_FragColor=vec4(col,1.);}`;
const sh=(t,s)=>{const o=g.createShader(t);g.shaderSource(o,s);g.compileShader(o);if(!g.getShaderParameter(o,g.COMPILE_STATUS))console.error(g.getShaderInfoLog(o));return o};
const pr=g.createProgram();g.attachShader(pr,sh(g.VERTEX_SHADER,'attribute vec2 a;varying vec2 v;void main(){v=a*.5+.5;gl_Position=vec4(a,0,1);}'));g.attachShader(pr,sh(g.FRAGMENT_SHADER,FS));g.linkProgram(pr);g.useProgram(pr);
g.bindBuffer(g.ARRAY_BUFFER,g.createBuffer());g.bufferData(g.ARRAY_BUFFER,new Float32Array([-1,-1,3,-1,-1,3]),g.STATIC_DRAW);
const al=g.getAttribLocation(pr,'a');g.enableVertexAttribArray(al);g.vertexAttribPointer(al,2,g.FLOAT,false,0,0);
g.bindTexture(g.TEXTURE_2D,g.createTexture());[g.TEXTURE_WRAP_S,g.TEXTURE_WRAP_T].forEach(p=>g.texParameteri(g.TEXTURE_2D,p,g.CLAMP_TO_EDGE));[g.TEXTURE_MIN_FILTER,g.TEXTURE_MAG_FILTER].forEach(p=>g.texParameteri(g.TEXTURE_2D,p,g.LINEAR));g.pixelStorei(g.UNPACK_FLIP_Y_WEBGL,true);
const U={};['R','t','m','mir','W','tint'].forEach(n=>U[n]=g.getUniformLocation(pr,n));
const C={f:'user',mir:true};
function gr(t,f,faces){g.viewport(0,0,glc.width,glc.height);g.texImage2D(g.TEXTURE_2D,0,g.RGBA,g.RGBA,g.UNSIGNED_BYTE,vid);
 const w=new Float32Array(16);let n=0;
 if(f.w)faces.forEach(F=>f.w.forEach(([a,r,s])=>{if(n<4){const p=F.A[a];w.set([p[0]/out.width,1-p[1]/out.height,r*F.W/out.height,s],n*4);n++}}));
 g.uniform2f(U.R,glc.width,glc.height);g.uniform1f(U.t,t);g.uniform1f(U.m,f.m||0);g.uniform1f(U.mir,C.mir?1:0);g.uniform4fv(U.W,w);g.uniform4fv(U.tint,f.tint||[1,1,1,0]);g.drawArrays(g.TRIANGLES,0,3)}
// ---------- Face helpers (MediaPipe 478-point landmarks) ----------
const OV=[10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109],LIPS=[61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146];
function mk(Lm,bs){const w=out.width,h=out.height,P=i=>[(C.mir?1-Lm[i].x:Lm[i].x)*w,Lm[i].y*h];
 let le=P(468),re=P(473),lc=P(234),rc=P(454);if(le[0]>re[0])[le,re]=[re,le];if(lc[0]>rc[0])[lc,rc]=[rc,lc];
 const W=Math.hypot(rc[0]-lc[0],rc[1]-lc[1]),r=Math.atan2(re[1]-le[1],re[0]-le[0]);
 const A={le,re,lc,rc,no:P(1),mo:P(13),ch:P(152),fh:P(10),ey:[(le[0]+re[0])/2,(le[1]+re[1])/2]};
 const j=bs&&bs.categories.find(c=>c.categoryName=='jawOpen');return {P,A,W,r,jaw:j?j.score:0}}
const L2=(F,a,dx=0,dy=0)=>{const c=Math.cos(F.r),s=Math.sin(F.r),p=F.A[a];return [p[0]+(dx*c-dy*s)*F.W,p[1]+(dx*s+dy*c)*F.W]};
const st=(F,e,a,s,dx=0,dy=0,rot=0)=>{const [x,y]=L2(F,a,dx,dy);ctx.save();ctx.translate(x,y);ctx.rotate(F.r+rot);ctx.font=s*F.W+'px serif';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(e,0,0);ctx.restore()};
const loc=(F,a,fn)=>{ctx.save();ctx.translate(...F.A[a]);ctx.rotate(F.r);ctx.scale(F.W,F.W);fn();ctx.restore()};
const poly=(F,idx)=>{ctx.beginPath();idx.forEach((i,k)=>{const [x,y]=F.P(i);k?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.closePath()};
const filt=(F,idx,css,al=1)=>{ctx.save();poly(F,idx);ctx.clip();ctx.filter=css;ctx.globalAlpha=al;ctx.drawImage(glc,0,0);ctx.restore()};
const PT=[];const em=(e,x,y,vx,vy,s,l,gv=0)=>{if(PT.length<400)PT.push({e,x,y,vx,vy,s,l,L:l,g:gv})};
function pp(dt){ctx.textAlign='center';ctx.textBaseline='middle';for(let i=PT.length-1;i>=0;i--){const p=PT[i];p.l-=dt;if(p.l<=0||p.y>out.height+60){PT.splice(i,1);continue}p.vy+=p.g*dt;p.x+=p.vx*dt;p.y+=p.vy*dt;ctx.globalAlpha=Math.min(1,p.l/p.L*2);ctx.font=p.s+'px serif';ctx.fillText(p.e,p.x,p.y)}ctx.globalAlpha=1}
const mesh=(F,col)=>{const T=FaceLandmarker.FACE_LANDMARKS_TESSELATION||[];ctx.strokeStyle=col;ctx.lineWidth=1;ctx.beginPath();T.forEach(c=>{ctx.moveTo(...F.P(c.start));ctx.lineTo(...F.P(c.end))});ctx.stroke()};
const glow=(F,a,col)=>{const [x,y]=F.A[a],r=F.W*.14,gd=ctx.createRadialGradient(x,y,0,x,y,r);gd.addColorStop(0,'#fff');gd.addColorStop(.3,col);gd.addColorStop(1,'#0000');ctx.fillStyle=gd;ctx.fillRect(x-r,y-r,r*2,r*2)};
const both=(fn)=>[-1,1].forEach(fn);
// ---------- AR filters ----------
const spider=(F,t)=>{for(let k=0;k<3;k++){const u=t*.5+k*2.1;st(F,'🕷️','no',.2,Math.cos(u)*.38,Math.sin(u*1.3)*.3+.1,Math.cos(u)*.8+Math.sin(t*25)*.12)}
 const [x,y]=L2(F,'fh',.15,.05+.06*Math.sin(t*2));ctx.strokeStyle='#fffa';ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,y);ctx.stroke();st(F,'🕷️','fh',.25,.15,.05+.06*Math.sin(t*2),Math.PI);
 st(F,'🕸️','fh',.6,-.4,.12);st(F,'🕸️','fh',.6,.4,.12)};
const oldAge=F=>{filt(F,OV,'grayscale(.7) contrast(1.4) brightness(.92) blur(.7px)');ctx.strokeStyle='rgba(60,40,30,.5)';ctx.lineWidth=.012;
 loc(F,'fh',()=>{for(let i=0;i<3;i++){ctx.beginPath();ctx.moveTo(-.25,.1+i*.05);ctx.quadraticCurveTo(0,.07+i*.05,.25,.1+i*.05);ctx.stroke()}ctx.fillStyle='rgba(235,235,235,.8)';ctx.beginPath();ctx.ellipse(0,-.02,.5,.24,0,Math.PI,0);ctx.fill()});
 [['le',-1],['re',1]].forEach(([a,s])=>loc(F,a,()=>{for(let i=-1;i<=1;i++){ctx.beginPath();ctx.moveTo(s*.1,i*.01);ctx.lineTo(s*.19,i*.05);ctx.stroke()}ctx.beginPath();ctx.arc(0,.1,.08,.2,2.9);ctx.stroke()}));
 loc(F,'no',()=>both(s=>{ctx.beginPath();ctx.moveTo(s*.1,0);ctx.quadraticCurveTo(s*.18,.1,s*.15,.2);ctx.stroke()}))};
const beauty=F=>{filt(F,OV,'blur(5px) brightness(1.1) saturate(1.1)',.65);ctx.fillStyle='rgba(255,70,110,.2)';loc(F,'no',()=>both(s=>{ctx.beginPath();ctx.ellipse(s*.28,.05,.13,.1,0,0,7);ctx.fill()}));poly(F,LIPS);ctx.fillStyle='rgba(215,25,70,.5)';ctx.fill()};
const skull=F=>{ctx.save();poly(F,OV);ctx.fillStyle='rgba(245,245,245,.8)';ctx.fill();ctx.restore();ctx.fillStyle='#000';['le','re'].forEach(a=>loc(F,a,()=>{ctx.beginPath();ctx.arc(0,0,.1,0,7);ctx.fill()}));
 loc(F,'no',()=>{ctx.beginPath();ctx.moveTo(0,-.04);ctx.lineTo(-.04,.05);ctx.lineTo(.04,.05);ctx.fill()});ctx.strokeStyle='#000';loc(F,'mo',()=>{ctx.lineWidth=.012;for(let i=-4;i<=4;i++){ctx.beginPath();ctx.moveTo(i*.035,-.04);ctx.lineTo(i*.035,.08);ctx.stroke()}})};
const ears=(F,fn)=>loc(F,'fh',()=>both(s=>{ctx.save();ctx.scale(s,1);fn();ctx.restore()}));
const ar=(i,n,fn,o)=>({c:'AR',i,n,ar:fn,...o}),fxf=(i,n,m,fx)=>({c:'Fun',i,n,m,fx}),C_=(i,n,m)=>({c:'Color',i,n,m}),wp=(i,n,w,tint)=>({c:'Warp',i,n,w,tint});
const FL=[
 ar('🕷️','Spider Crawl',spider),
 ar('🧓','Old Age',oldAge),ar('💄','Beauty',beauty),ar('💀','Skull Paint',skull),
 ar('🕶️','Shades',F=>st(F,'🕶️','ey',1.05)),ar('👓','Nerd',F=>st(F,'👓','ey',1.05)),
 ar('😍','Heart Eyes',(F,t)=>{const s=.32+.05*Math.sin(t*8);st(F,'❤️','le',s);st(F,'❤️','re',s)}),
 ar('🤩','Star Eyes',(F,t)=>{st(F,'⭐','le',.3,0,0,t);st(F,'⭐','re',.3,0,0,-t)}),
 ar('👑','Crown',F=>st(F,'👑','fh',.6,0,-.12)),ar('🎩','Top Hat',F=>st(F,'🎩','fh',.8,0,-.22)),ar('🧢','Cap',F=>st(F,'🧢','fh',.8,0,-.12)),ar('🎓','Grad',F=>st(F,'🎓','fh',.8,0,-.15)),ar('🎀','Bow',F=>st(F,'🎀','fh',.4,.3,-.12,.4)),
 ar('🌸','Flower Crown',F=>[-.36,-.18,0,.18,.36].forEach((x,i)=>st(F,['🌸','🌼','🌺','🌷','🌻'][i],'fh',.22,x,-.08,x))),
 ar('😈','Devil',F=>{ctx.fillStyle='#c00';loc(F,'fh',()=>both(s=>{ctx.beginPath();ctx.moveTo(s*.28,.02);ctx.quadraticCurveTo(s*.42,-.12,s*.34,-.3);ctx.quadraticCurveTo(s*.2,-.1,s*.12,.02);ctx.fill()}))},{tint:[1.3,.7,.7,1]}),
 ar('😇','Angel',(F,t)=>{ctx.save();ctx.shadowColor='#ff0';ctx.shadowBlur=25;loc(F,'fh',()=>{ctx.strokeStyle='#ffd700';ctx.lineWidth=.04;ctx.beginPath();ctx.ellipse(0,-.2+.01*Math.sin(t*3),.28,.07,0,0,7);ctx.stroke()});ctx.restore()}),
 ar('🐱','Cat',F=>{ears(F,()=>{ctx.fillStyle='#333';ctx.beginPath();ctx.moveTo(.15,.05);ctx.lineTo(.33,-.32);ctx.lineTo(.42,.08);ctx.fill();ctx.fillStyle='#f9a';ctx.beginPath();ctx.moveTo(.2,.03);ctx.lineTo(.32,-.2);ctx.lineTo(.37,.05);ctx.fill()});loc(F,'no',()=>{ctx.fillStyle='#f6a';ctx.beginPath();ctx.moveTo(-.04,-.02);ctx.lineTo(.04,-.02);ctx.lineTo(0,.03);ctx.fill();ctx.strokeStyle='#fff';ctx.lineWidth=.008;both(s=>[-.03,0,.03].forEach(d=>{ctx.beginPath();ctx.moveTo(s*.1,d);ctx.lineTo(s*.5,d*3);ctx.stroke()}))})}),
 ar('🐶','Dog',(F,t)=>{ears(F,()=>{ctx.fillStyle='#7a4a21';ctx.beginPath();ctx.ellipse(.42,.1,.1,.24,.3,0,7);ctx.fill()});loc(F,'no',()=>{ctx.fillStyle='#111';ctx.beginPath();ctx.ellipse(0,.02,.06,.04,0,0,7);ctx.fill()});if(F.jaw>.2)st(F,'👅','mo',.28,0,.14+.01*Math.sin(t*12))}),
 ar('🐰','Bunny',F=>ears(F,()=>{ctx.fillStyle='#fff';ctx.beginPath();ctx.ellipse(.15,-.4,.09,.3,.1,0,7);ctx.fill();ctx.fillStyle='#f9b';ctx.beginPath();ctx.ellipse(.15,-.4,.04,.22,.1,0,7);ctx.fill()})),
 ar('🥸','Mustache',F=>loc(F,'no',()=>{ctx.fillStyle='#111';both(s=>{ctx.save();ctx.scale(s,1);ctx.beginPath();ctx.moveTo(0,.07);ctx.bezierCurveTo(.08,.02,.2,.03,.28,.13);ctx.bezierCurveTo(.2,.08,.1,.12,0,.11);ctx.fill();ctx.restore()})})),
 ar('🏴‍☠️','Pirate',F=>{loc(F,'re',()=>{ctx.fillStyle='#111';ctx.strokeStyle='#111';ctx.lineWidth=.015;ctx.beginPath();ctx.ellipse(0,0,.11,.09,0,0,7);ctx.fill();ctx.beginPath();ctx.moveTo(-.1,-.04);ctx.lineTo(-.5,-.2);ctx.stroke()});loc(F,'fh',()=>{ctx.fillStyle='#c11';ctx.fillRect(-.5,-.04,1,.13)})}),
 ar('🤡','Clown',F=>{st(F,'🔴','no',.2);loc(F,'fh',()=>both(s=>{ctx.fillStyle=s<0?'#f44':'#39f';ctx.beginPath();ctx.arc(s*.42,0,.16,0,7);ctx.fill()}))}),
 ar('🧛','Vampire',F=>{loc(F,'mo',()=>{ctx.fillStyle='#fff';both(s=>{ctx.beginPath();ctx.moveTo(s*.07,-.01);ctx.lineTo(s*.11,-.01);ctx.lineTo(s*.09,.07);ctx.fill()})});st(F,'🩸','mo',.12,.1,.12)},{tint:[.9,.85,1,1]}),
 ar('🧟','Zombie',F=>{st(F,'🧠','fh',.3,.1,-.05);st(F,'🩸','lc',.2,.1,.1)},{tint:[.7,1.1,.6,1]}),
 ar('👽','Alien',F=>{ctx.strokeStyle='#4f4';ctx.lineWidth=3;both(s=>{const a=L2(F,'fh',s*.15,0),b=L2(F,'fh',s*.3,-.35);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();ctx.fillStyle='#4f4';ctx.beginPath();ctx.arc(...b,F.W*.04,0,7);ctx.fill()})},{w:[['le',.2,.7],['re',.2,.7],['ch',.35,-.4]],tint:[.6,1.2,.6,1]}),
 ar('🤖','Robot',F=>{mesh(F,'rgba(0,255,255,.45)');glow(F,'le','#0ff');glow(F,'re','#0ff')},{tint:[.8,.95,1.2,1]}),
 ar('🕸️','Face Mesh',F=>mesh(F,'rgba(255,255,255,.6)')),
 ar('👁️','Red Eyes',F=>{glow(F,'le','#f00');glow(F,'re','#f00')},{tint:[1,.8,.8,1]}),
 ar('🌈','Rainbow',F=>{if(F.jaw>.2)loc(F,'mo',()=>{ctx.globalAlpha=Math.min(1,F.jaw*3);['#f00','#f80','#ff0','#0c0','#08f','#80f'].forEach((c,i)=>{ctx.fillStyle=c;ctx.fillRect(-.12+i*.04,0,.04,4)});ctx.globalAlpha=1})}),
 ar('🐉','Dragon Fire',F=>{if(F.jaw>.2)for(let i=0;i<3;i++){const [x,y]=L2(F,'mo'),v=250+rnd()*250;em('🔥',x+(rnd()-.5)*F.W*.1,y,-Math.sin(F.r)*v+(rnd()-.5)*90,Math.cos(F.r)*v,F.W*(.15+rnd()*.2),.7)}}),
 ar('😭','Tears',(F,t)=>['le','re'].forEach((a,k)=>{for(let i=0;i<3;i++){const u=(t*.7+i/3+k*.3)%1;ctx.globalAlpha=1-u;st(F,'💧',a,.14,(k?.06:-.06),.1+u*.5)}ctx.globalAlpha=1})),
 ar('💖','Love',F=>{if(rnd()<.25){const [x,y]=L2(F,'fh',(rnd()-.5)*1.2,-.1+rnd()*.3);em(['💖','💕','💗'][rnd()*3|0],x,y,(rnd()-.5)*50,-80-rnd()*60,F.W*.15,1.6)}}),
 ar('✨','Sparkle',F=>{if(rnd()<.5){const [x,y]=L2(F,'no',(rnd()-.5)*1.1,(rnd()-.4)*1.2);em('✨',x,y,0,-20,F.W*(.08+rnd()*.12),.8)}}),
 ar('🎭','Mask',F=>st(F,'🎭','no',1.3,0,-.05)),
 wp('👀','Big Eyes',[['le',.22,.5],['re',.22,.5]]),wp('🐞','Bug Eyes',[['le',.28,.85],['re',.28,.85]]),wp('👄','Big Mouth',[['mo',.3,.6]]),wp('👃','Big Nose',[['no',.25,.65]]),
 wp('🐹','Chubby',[['lc',.4,.4],['rc',.4,.4]]),wp('🐭','Tiny Face',[['no',.8,-.35]]),wp('🥜','Pointy Chin',[['ch',.3,-.5]]),wp('🫥','Squish',[['ch',.5,-.5],['fh',.5,-.3]]),
 C_('✨','Normal',0),C_('⬛','B&W',1),C_('📜','Sepia',2),C_('📼','Vintage',3),C_('🎞️','Noir',4),C_('🧊','Cool',5),C_('🌅','Warm',6),C_('💡','Neon',7),C_('🔥','Thermal',8),C_('🌙','Night Vision',9),C_('👾','Glitch',10),C_('📺','VHS',11),C_('🔲','Pixel',12),C_('💥','Comic',13),C_('✏️','Sketch',14),C_('⚫','Halftone',15),C_('☁️','Dreamy',16),C_('🟣','Duotone',17),C_('🌆','Cyberpunk',18),C_('🟩','Matrix',19),C_('🔄','Invert',20),C_('🩻','X-Ray',21),C_('🌈','RGB Split',22),C_('🌊','Underwater',23),C_('🍺','Drunk',24),C_('🎨','Pop Art',34),C_('🌇','Golden Hour',35),C_('💫','Glow',33),
 {c:'Fun',i:'🪞',n:'Mirror',m:25},{c:'Fun',i:'🔮',n:'Kaleido',m:26},{c:'Fun',i:'🌀',n:'Swirl',m:27},{c:'Fun',i:'🐟',n:'Fisheye',m:28},{c:'Fun',i:'💓',n:'Pulse',m:30},{c:'Fun',i:'🫠',n:'Pinch',m:31},{c:'Fun',i:'〰️',n:'Wave',m:32},
 fxf('🎉','Confetti',0,()=>{for(let i=0;i<2;i++)em(['🎉','🎊','✨','⭐'][rnd()*4|0],rnd()*out.width,-20,(rnd()-.5)*80,120+rnd()*200,30,7,100)}),
 fxf('❄️','Snow',0,()=>{if(rnd()<.5)em('❄️',rnd()*out.width,-20,(rnd()-.5)*40,80+rnd()*80,16+rnd()*22,14)}),
 fxf('💖','Love Rain',0,()=>{if(rnd()<.3)em(['💖','💕'][rnd()*2|0],rnd()*out.width,-20,0,100+rnd()*100,30,10)}),
 fxf('🪩','Disco',34,()=>{if(rnd()<.4)em(['⭐','✨','🎵','🎶'][rnd()*4|0],rnd()*out.width,rnd()*out.height,0,-30,30,1.2)})
];
// ---------- main loop ----------
let cur=FL[0],det,last=-1,faces=[],t0=performance.now(),prev=0,stream;
const loop=()=>{requestAnimationFrame(loop);if(vid.readyState<2)return;const w=vid.videoWidth,h=vid.videoHeight;
 if(out.width!=w||out.height!=h){out.width=glc.width=w;out.height=glc.height=h}
 const t=(performance.now()-t0)/1000,dt=Math.min(.1,t-prev);prev=t;
 if(det&&vid.currentTime!=last&&(cur.ar||cur.w)){last=vid.currentTime;try{const r=det.detectForVideo(vid,performance.now());faces=(r.faceLandmarks||[]).map((l,i)=>mk(l,r.faceBlendshapes&&r.faceBlendshapes[i]))}catch(e){}}
 const fc=cur.ar||cur.w?faces:[];gr(t,cur,fc);ctx.drawImage(glc,0,0);
 ctx.fillStyle='#fff';if(cur.ar)fc.forEach(F=>cur.ar(F,t));if(cur.fx)cur.fx(t,dt);pp(dt)};
// ---------- UI ----------
const tabs=['AR','Warp','Color','Fun'];let tab='AR';const tb=$('#tb'),sl=$('#st');
const show=()=>{tb.innerHTML=sl.innerHTML='';tabs.forEach(n=>{const b=document.createElement('button');b.className='t'+(n==tab?' on':'');b.textContent=n;b.onclick=()=>{tab=n;show()};tb.append(b)});
 FL.filter(f=>f.c==tab).forEach(f=>{const b=document.createElement('button');b.className='f'+(f==cur?' on':'');b.innerHTML=`<i>${f.i}</i><s>${f.n}</s>`;b.onclick=()=>{cur=f;PT.length=0;sl.querySelectorAll('.f').forEach(x=>x.classList.remove('on'));b.classList.add('on')};sl.append(b)})};
const toast=s=>{const m=$('#m');m.textContent=s;m.style.display='block';setTimeout(()=>m.style.display='none',2500)};
const save=(b,x)=>{const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download=`filterz-${Date.now()}.${x}`;a.click();toast('Saved to Downloads')};
async function cam(){try{stream&&stream.getTracks().forEach(t=>t.stop());stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:C.f,width:{ideal:1280},height:{ideal:720}},audio:false});vid.srcObject=stream;await vid.play()}catch(e){toast('Camera error: '+e.message)}}
$('#fl').onclick=()=>{C.f=C.f=='user'?'environment':'user';C.mir=C.f=='user';faces=[];cam()};
let rec,chunks=[],tm,on=0;const sb=$('#s');
sb.onpointerdown=()=>{tm=setTimeout(async()=>{tm=0;on=1;sb.classList.add('r');const s=out.captureStream(30);try{const a=await navigator.mediaDevices.getUserMedia({audio:true});a.getAudioTracks().forEach(x=>s.addTrack(x))}catch(e){}
 if(!on)return;const mt=['video/mp4;codecs=avc1','video/webm;codecs=vp9','video/webm'].find(m=>MediaRecorder.isTypeSupported(m));rec=new MediaRecorder(s,{mimeType:mt});chunks=[];rec.ondataavailable=e=>chunks.push(e.data);rec.onstop=()=>save(new Blob(chunks,{type:mt}),mt.includes('mp4')?'mp4':'webm');rec.start()},400)};
sb.onpointerup=sb.onpointercancel=()=>{if(tm){clearTimeout(tm);tm=0;out.toBlob(b=>save(b,'jpg'),'image/jpeg',.95)}else if(on){on=0;sb.classList.remove('r');rec&&rec.state!='inactive'&&rec.stop()}};
show();cam().then(()=>loop());
if('serviceWorker'in navigator)navigator.serviceWorker.register('sw.js').catch(()=>{});
(async()=>{try{const f=await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm');const o=d=>({baseOptions:{modelAssetPath:'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',delegate:d},runningMode:'VIDEO',numFaces:2,outputFaceBlendshapes:true});
 try{det=await FaceLandmarker.createFromOptions(f,o('GPU'))}catch(e){det=await FaceLandmarker.createFromOptions(f,o('CPU'))}$('#m').style.display='none'}catch(e){toast('Face tracking failed to load (check internet). Color filters still work.')}})();
