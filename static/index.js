async function fetchAsset(assetPath) {
    const normalizedPath = String(assetPath || "").replace(/^\/+/, "")
    const encodedPath = normalizedPath
        .split("/")
        .map(encodeURIComponent)
        .join("/")

    const response = await fetch(`/assets/${encodedPath}`)
    if (!response.ok) {
        throw new Error(`Failed to fetch asset: ${normalizedPath} (${response.status})`)
    }

    return await response.blob()
}

async function fetchAllAssets() {
    const [pipe, bird, bg] = await Promise.all([
        fetchAsset("pipe-green.png"),
        fetchAsset("yellowbird-midflap.png"),
        fetchAsset("bg.png"),
    ])

    return { pipe, bird, bg }
}

function blobToImage(blob) {
    return new Promise((resolve, reject) => {
        const objectUrl = URL.createObjectURL(blob)
        const image = new Image()

        image.onload = () => {
            URL.revokeObjectURL(objectUrl)
            resolve(image)
        }
        image.onerror = (error) => {
            URL.revokeObjectURL(objectUrl)
            reject(error)
        }

        image.src = objectUrl
    })
}

function intersects(a, b) {
    return (
        a.x < b.x + b.width &&
        a.x + a.width > b.x &&
        a.y < b.y + b.height &&
        a.y + a.height > b.y
    )
}

function getImageAlphaMask(image) {
    const canvas = document.createElement("canvas");
    canvas.width = image.width;
    canvas.height = image.height;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) throw new Error("Could not create mask context");
    ctx.drawImage(image, 0, 0);
    const imageData = ctx.getImageData(0, 0, image.width, image.height).data;
    const mask = new Uint8Array(image.width * image.height);
    for (let i = 0; i < mask.length; i++) {
        mask[i] = imageData[i * 4 + 3] > 0 ? 1 : 0;
    }
    return mask;
}

function getRotatedImageAlphaMask(image) {
    const canvas = document.createElement("canvas");
    canvas.width = image.width;
    canvas.height = image.height;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) throw new Error("Could not create mask context");

    ctx.translate(canvas.width / 2, canvas.height / 2);
    ctx.rotate(Math.PI);
    ctx.drawImage(image, -image.width / 2, -image.height / 2);

    const imageData = ctx.getImageData(0, 0, image.width, image.height).data;
    const mask = new Uint8Array(image.width * image.height);
    for (let i = 0; i < mask.length; i++) {
        mask[i] = imageData[i * 4 + 3] > 0 ? 1 : 0;
    }
    return mask;
}

function pixelCollide(rectA, maskA, rectB, maskB) {
    const xMin = Math.max(Math.floor(rectA.x), Math.floor(rectB.x));
    const xMax = Math.min(Math.floor(rectA.x + rectA.width), Math.floor(rectB.x + rectB.width));
    const yMin = Math.max(Math.floor(rectA.y), Math.floor(rectB.y));
    const yMax = Math.min(Math.floor(rectA.y + rectA.height), Math.floor(rectB.y + rectB.height));

    if (xMin >= xMax || yMin >= yMax) return false;

    for (let y = yMin; y < yMax; y++) {
        for (let x = xMin; x < xMax; x++) {
            const aX = x - Math.floor(rectA.x);
            const aY = y - Math.floor(rectA.y);
            const bX = x - Math.floor(rectB.x);
            const bY = y - Math.floor(rectB.y);

            if (maskA[aY * rectA.width + aX] && maskB[bY * rectB.width + bX]) {
                return true;
            }
        }
    }
    return false;
}

function createPredictSocket() {
    const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:"
    const ws = new WebSocket(`${wsProtocol}//${window.location.host}/predict`)

    ws.addEventListener("open", () => {
        console.log("/predict websocket connected")
    })

    ws.addEventListener("error", (error) => {
        console.error("/predict websocket error:", error)
    })

    ws.addEventListener("close", () => {
        console.log("/predict websocket closed")
    })

    return ws
}

function captureFrame84AreaAverage(canvas, captureCtx) {
    // 1. Get full-resolution pixel data from the source canvas
    const sw = canvas.width;
    const sh = canvas.height;

    // We reuse captureCtx just to read the full image data if it was drawn there.
    // Wait, the easiest way to read the full source canvas is to grab its imageData directly.
    const sourceCtx = canvas.getContext("2d", { willReadFrequently: true });
    if (!sourceCtx) throw new Error("Could not get source context");

    const sourceImageData = sourceCtx.getImageData(0, 0, sw, sh).data;

    const dw = 84;
    const dh = 84;

    const scaleX = dw / sw;
    const scaleY = dh / sh;

    // 2. We will output an 84x84 array of grayscale values
    const frame = [];

    for (let dy = 0; dy < dh; dy++) {
        const row = [];
        // Determine the horizontal mapped region [sy1, sy2]
        const sy1 = dy / scaleY;
        const sy2 = (dy + 1) / scaleY;

        for (let dx = 0; dx < dw; dx++) {
            const sx1 = dx / scaleX;
            const sx2 = (dx + 1) / scaleX;

            let sum_gray = 0;
            let total_area = 0;

            // Iterate over all integer pixels in the source bounding box
            const startX = Math.floor(sx1);
            const endX = Math.min(Math.ceil(sx2), sw);
            const startY = Math.floor(sy1);
            const endY = Math.min(Math.ceil(sy2), sh);

            for (let sy = startY; sy < endY; sy++) {
                // Determine vertical overlap
                let yOverlap = 1.0;
                if (sy < sy1) {
                    yOverlap = (sy + 1) - sy1;
                } else if (sy + 1 > sy2) {
                    yOverlap = sy2 - sy;
                }

                for (let sx = startX; sx < endX; sx++) {
                    // Determine horizontal overlap
                    let xOverlap = 1.0;
                    if (sx < sx1) {
                        xOverlap = (sx + 1) - sx1;
                    } else if (sx + 1 > sx2) {
                        xOverlap = sx2 - sx;
                    }

                    const area = xOverlap * yOverlap;
                    total_area += area;

                    const idx = (sy * sw + sx) * 4;
                    const r = sourceImageData[idx];
                    const g = sourceImageData[idx + 1];
                    const b = sourceImageData[idx + 2];

                    // Standard grayscale formula matching OpenCV
                    const gray = 0.299 * r + 0.587 * g + 0.114 * b;

                    sum_gray += gray * area;
                }
            }

            // Average it out and round to nearest integer
            let final_gray = 0;
            if (total_area > 0) {
                final_gray = Math.round(sum_gray / total_area);
            }
            row.push(final_gray);
        }
        frame.push(row);
    }

    return frame;
}

class FlappyBirdCanvasGame {
    constructor(canvas, ctx, images) {
        this.canvas = canvas
        this.ctx = ctx
        this.images = images

        this.width = images.bg.width
        this.height = images.bg.height

        this.gravity = 0.25
        this.flapStrength = -5
        this.pipeVx = 4
        this.distBetweenPairs = 200
        this.interDist = 100
        this.gapDelta = 150
        this.groundTop = this.height
        this.minGapTopClearance = 90
        this.minGapBottomClearance = 85

        this.bird = {
            x: 50 - Math.floor(images.bird.width / 2),
            y: 256 - Math.floor(images.bird.height / 2),
            width: images.bird.width,
            height: images.bird.height,
            velocity: 0,
        }

        this.birdMask = getImageAlphaMask(images.bird)
        this.pipeMaskBottom = getImageAlphaMask(images.pipe)
        this.pipeMaskTop = getRotatedImageAlphaMask(images.pipe)

        this.pipes = []
        this.lastGapY = null
        this.flapQueued = false

        this.canvas.width = this.width
        this.canvas.height = this.height

        this.gameplayCanvas = document.createElement("canvas")
        this.gameplayCanvas.width = this.width
        this.gameplayCanvas.height = this.height
        const gameplayCtx = this.gameplayCanvas.getContext("2d")
        if (!gameplayCtx) {
            throw new Error("Could not create gameplay context")
        }
        this.gameplayCtx = gameplayCtx

        this.reset()
    }

    randomInt(min, max) {
        return Math.floor(Math.random() * (max - min + 1)) + min
    }

    reset() {
        this.bird.x = 50 - Math.floor(this.images.bird.width / 2)
        this.bird.y = 256 - Math.floor(this.images.bird.height / 2)
        this.bird.velocity = 0

        this.pipes = []
        this.score = 0
        this.lastGapY = null

        for (let i = 0; i < 10; i += 1) {
            this.generatePipePair()
        }
    }

    generatePipePair() {
        const halfGap = this.interDist / 2
        // Limit gap bounds so we don't expose the top or bottom of the 320px pipe image
        const groundTop = this.height - 112
        const minY = Math.max(100, halfGap + groundTop - this.images.pipe.height)
        const maxY = Math.min(this.height - 100, this.images.pipe.height - halfGap)

        let gapCenterY
        if (this.lastGapY === null) {
            gapCenterY = this.randomInt(minY, maxY)
        } else {
            const lowBound = Math.max(minY, this.lastGapY - this.gapDelta)
            const highBound = Math.min(maxY, this.lastGapY + this.gapDelta)
            gapCenterY = this.randomInt(lowBound, highBound)
        }

        this.lastGapY = gapCenterY

        const x =
            this.pipes.length > 0
                ? this.pipes[this.pipes.length - 1].x + this.distBetweenPairs
                : this.width - Math.floor(this.images.pipe.width / 2)

        const topBottom = gapCenterY - halfGap
        const bottomTop = gapCenterY + halfGap

        this.pipes.push({
            x,
            topY: topBottom - this.images.pipe.height,
            bottomY: bottomTop,
            scored: false,
        })
    }

    checkCollision() {
        const groundTop = this.height - 112
        if (this.bird.y + this.bird.height >= groundTop || this.bird.y <= 0) {
            return true
        }

        const birdRect = {
            x: this.bird.x,
            y: this.bird.y,
            width: this.bird.width,
            height: this.bird.height,
        }

        for (const pipe of this.pipes) {
            const topRect = {
                x: pipe.x,
                y: pipe.topY,
                width: this.images.pipe.width,
                height: this.images.pipe.height,
            }
            const bottomRect = {
                x: pipe.x,
                y: pipe.bottomY,
                width: this.images.pipe.width,
                height: this.images.pipe.height,
            }

            if (intersects(birdRect, topRect)) {
                if (pixelCollide(birdRect, this.birdMask, topRect, this.pipeMaskTop)) {
                    return true
                }
            }
            if (intersects(birdRect, bottomRect)) {
                if (pixelCollide(birdRect, this.birdMask, bottomRect, this.pipeMaskBottom)) {
                    return true
                }
            }
        }

        return false
    }

    step() {
        let passedPipe = false

        for (let i = this.pipes.length - 1; i >= 0; i -= 1) {
            const pipe = this.pipes[i]
            pipe.x -= this.pipeVx

            if (this.bird.x > pipe.x + this.images.pipe.width && !pipe.scored) {
                passedPipe = true
                pipe.scored = true
            }

            if (pipe.x + this.images.pipe.width < 0) {
                this.pipes.splice(i, 1)
            }
        }

        if (passedPipe) {
            this.score += 1
            this.generatePipePair()
        }

        this.bird.velocity += this.gravity
        if (this.flapQueued) {
            this.bird.velocity = this.flapStrength
        }
        this.flapQueued = false

        this.bird.y += Math.trunc(this.bird.velocity)

        return this.checkCollision()
    }

    renderGameplay(targetCtx) {
        targetCtx.clearRect(0, 0, this.width, this.height)
        targetCtx.drawImage(this.images.bg, 0, 0)

        for (const pipe of this.pipes) {
            targetCtx.drawImage(this.images.pipe, pipe.x, pipe.bottomY)

            targetCtx.save()
            targetCtx.translate(
                pipe.x + this.images.pipe.width,
                pipe.topY + this.images.pipe.height
            )
            targetCtx.rotate(Math.PI)
            targetCtx.drawImage(this.images.pipe, 0, 0)
            targetCtx.restore()
        }

        targetCtx.drawImage(this.images.bird, this.bird.x, this.bird.y)
    }

    getObservationCanvas() {
        return this.gameplayCanvas
    }

    render() {
        this.renderGameplay(this.gameplayCtx)

        this.ctx.clearRect(0, 0, this.width, this.height)
        this.ctx.drawImage(this.gameplayCanvas, 0, 0)
    }
}

async function displayOnCanvas() {
    const canvas = document.getElementById("main-canvas")
    if (!canvas) {
        throw new Error("Canvas with id 'main-canvas' not found")
    }

    const ctx = canvas.getContext("2d")
    if (!ctx) {
        throw new Error("2D canvas context is not available")
    }

    const assets = await fetchAllAssets()
    const [bgImage, pipeImage, birdImage] = await Promise.all([
        blobToImage(assets.bg),
        blobToImage(assets.pipe),
        blobToImage(assets.bird),
    ])

    const game = new FlappyBirdCanvasGame(canvas, ctx, {
        bg: bgImage,
        pipe: pipeImage,
        bird: birdImage,
    })

    let pendingAction = 0
    let actionNeeded = true
    let waitingForAction = false

    const predictSocket = createPredictSocket()

    const captureCanvas = document.createElement("canvas")
    captureCanvas.width = 84
    captureCanvas.height = 84
    const captureCtx = captureCanvas.getContext("2d", { willReadFrequently: true })
    if (!captureCtx) {
        throw new Error("Could not create frame capture context")
    }

    // Load ONNX Model
    console.log("Loading ONNX model...");
    const session = await ort.InferenceSession.create('/static/flappy_bird.onnx');
    console.log("Model loaded successfully!");

    // Frame Queue for stacking 4 frames
    const frameQueue = [];

    let lastFrameTime = performance.now()
    const FPS = 20
    const frameInterval = 1000 / FPS
    let accumulator = 0
    let gameCount = 1

    const loop = (now) => {
        const delta = now - lastFrameTime
        lastFrameTime = now
        accumulator += delta

        while (accumulator >= frameInterval) {
            if (actionNeeded && !waitingForAction) {
                game.renderGameplay(game.gameplayCtx)
                const frame = captureFrame84AreaAverage(game.getObservationCanvas(), captureCtx)

                if (frameQueue.length === 0) {
                    for (let i = 0; i < 4; i++) {
                        frameQueue.push(frame);
                    }
                } else {
                    frameQueue.push(frame);
                    if (frameQueue.length > 4) {
                        frameQueue.shift();
                    }
                }

                waitingForAction = true;

                // Create Tensor [1, 4, 84, 84]
                const tensorData = new Float32Array(4 * 84 * 84);
                let offset = 0;
                for (let c = 0; c < 4; c++) {
                    const cFrame = frameQueue[c];
                    for (let y = 0; y < 84; y++) {
                        for (let x = 0; x < 84; x++) {
                            tensorData[offset++] = cFrame[y][x];
                        }
                    }
                }

                const tensor = new ort.Tensor('float32', tensorData, [1, 4, 84, 84]);

                // Set initial property to ensure feed names match model inputs
                session.run({ input: tensor }).then((results) => {
                    const output = results.output.data;

                    // The ONNX model exported from MLflow direct policy predict wraps the argmax.
                    // The output is a single value array containing the predicted action (0 or 1).
                    const predictedAction = Number(output[0]);

                    pendingAction = predictedAction === 1 ? 1 : 0;
                    waitingForAction = false;
                    actionNeeded = false;
                    stepsRemaining = 4;
                }).catch((error) => {
                    console.error("ONNX Inference Error:", error);
                    waitingForAction = false;
                    actionNeeded = false;
                    stepsRemaining = 4;
                });
            }

            if (!actionNeeded && !waitingForAction) {
                let done = false;
                for (let i = 0; i < 4; i++) {
                    if (i === 0 && pendingAction === 1) {
                        game.flapQueued = true
                    } else {
                        game.flapQueued = false
                    }

                    done = game.step()
                    if (done) break
                }

                if (done) {
                    gameCount += 1
                    const finalScore = game.score
                    game.reset()
                    frameQueue.length = 0 // Clear frames on reset

                    if (predictSocket.readyState === WebSocket.OPEN) {
                        predictSocket.send(JSON.stringify({ reset: true, score: finalScore }))
                    }
                    actionNeeded = true
                    waitingForAction = false
                } else {
                    actionNeeded = true
                    waitingForAction = false
                }

                game.renderGameplay(game.gameplayCtx)
            }

            accumulator -= frameInterval
        }

        game.render()
        requestAnimationFrame(loop)
    }

    requestAnimationFrame(loop)
}

displayOnCanvas().catch((error) => {
    console.error("Failed to render assets on canvas:", error)
})
