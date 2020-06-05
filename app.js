const tf = require('@tensorflow/tfjs');
const bodyPix = require('@tensorflow-models/body-pix');

const Sel = x => document.querySelector(x);
const $video = Sel('video');
const $prompt = Sel('#prompt');
const $display = Sel('#display');
const $render = Sel('#render');

const $imgs = [];
for (let i = 0; i < 250; i += 10) {
    const $img = document.createElement('img');
    $img.src = '/laughing/frame-' + i + '.png';
    $imgs.push($img);
}

// ui
const $cons = document.getElementById('console');
function log(...args) {
    console.log(...args);
    $prompt.innerHTML += args.map(x => x ? x.toString() : 'null').join(' ') + '\n';
}
log('app.js loaded');

let net = null;
let playing = false;
let lastRender = performance.now();
let w = 0, h = 0;
let frame = 0;

async function start() {
    if (!('mediaDevices' in navigator && 'getUserMedia' in navigator.mediaDevices)) {
        throw 'camera not available in this browser';
    }

    const devices = await navigator.mediaDevices.enumerateDevices();
    devices.forEach(d => log('dev:', d.kind, d.label));
    console.log('devices:', devices);

    log('requesting video');
    let reqW = 480, reqH = 854;
    if (navigator.userAgent.match(/iPhone/i)) {
        [reqW, reqH] = [reqH, reqW];
    }
    const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
            facingMode: 'environment',
            width: reqW,
            height: reqH,
        },
    });
    log('got video:', stream);
    stream.getTracks().forEach(t => log('track', t.kind, t.label));

    $video.srcObject = stream;
    await new Promise(res => $video.onloadedmetadata = () => res());
    w = $video.videoWidth;
    h = $video.videoHeight;
    log('w:', w, 'h:', h);

    [$video, $render, $display].forEach(x => {
        x.width = w;
        x.height = h;
    });

    log('loading net...');
    net = await bodyPix.load({
        architecture: 'MobileNetV1',
        outputStride: 16,
        multiplier: 0.5,
        quantBytes: 2,
    });
    log('done');
    log('using backend', tf.getBackend());

    playing = true;
    $video.play();
    render();
    log('started');
};

async function render() {
    const ctx = $render.getContext('2d');

    ctx.clearRect(0, 0, w, h);
    ctx.drawImage($video, 0, 0, w, h);

    let seg = null;
    try {
        seg = await net.segmentPersonParts($render, {
            internalResolution: 'medium',
            maxDetections: 10,
        });
    } catch (e) {
        log('net err:', e);
        return;
    }
    // console.log('seg', seg);

    const bytes = ctx.getImageData(0, 0, w, h).data;
    // const bytes = new Uint8ClampedArray(w*h*4);
    for (let i = 0; i < w*h; i++) {
        const j = i*4;
        const label = seg.data[i];
        if (label === 0 || label === 1) {
            bytes[j+0] = 255;
            bytes[j+1] = 255;
            bytes[j+2] = 255;
            bytes[j+3] = 255;
        }
    }
    ctx.putImageData(new ImageData(bytes, w, h), 0, 0);

    const $img = $imgs[frame++ % $imgs.length];
    seg.allPoses.forEach(({ score, keypoints }) => {
        if (score < 0.2) return;
        const kps = keypoints.reduce((acc, val) => ({ [val.part]: val.position, ...acc }), {});
        const { leftEye, rightEye, nose } = kps;

        const a = rightEye.x - leftEye.x;
        const b = rightEye.y - leftEye.y;
        const sz = 5 * Math.sqrt(a*a+b*b);
        ctx.drawImage($img, nose.x-sz/2, nose.y-sz/2, sz, sz);
    });

    const dispCtx = $display.getContext('2d');
    dispCtx.clearRect(0, 0, w, h);
    dispCtx.drawImage($render, 0, 0);

    const now = performance.now();
    const fps = Math.round(1000/(now - lastRender)) + ' fps';
    // log('fps:', fps);
    Sel('#fps').innerHTML = fps;
    lastRender = now;
    if (playing) requestAnimationFrame(render);
}

$prompt.onclick = async () => {
    log('initial start');
    try {
        $prompt.innerHTML = 'loading...';
        await start();
        $prompt.style.display = 'none';
    } catch (e) {
        $prompt.innerHTML = e;
    }
};

$display.onclick = async () => {
    if (playing) {
        $video.pause();
    } else {
        $video.play();
        render();
    }
    playing = !playing;
};
