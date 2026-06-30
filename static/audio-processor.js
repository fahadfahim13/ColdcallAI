// AudioWorkletProcessor — captures mic audio and sends 8 kHz 16-bit mono PCM
// to the main thread via port.postMessage().
//
// The AudioContext runs at 16 kHz (set in index.html to match InWorld TTS output),
// so sampleRate here is 16000. Decimation ratio = 16000 / 8000 = 2.
// If the AudioContext rate ever changes, the ratio adjusts automatically.

class PCMProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this._ratio = Math.round(sampleRate / 8000);
    }

    process(inputs) {
        const channel = inputs[0] && inputs[0][0];
        if (!channel) return true;

        const out = [];
        for (let i = 0; i < channel.length; i += this._ratio) {
            const s = Math.max(-1, Math.min(1, channel[i]));
            out.push(s < 0 ? s * 0x8000 : s * 0x7FFF);
        }

        if (out.length) {
            const buf = new Int16Array(out);
            this.port.postMessage(buf.buffer, [buf.buffer]);
        }

        return true;
    }
}

registerProcessor('pcm-processor', PCMProcessor);
