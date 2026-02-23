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

function captureFrame84(canvas, captureCtx) {
    captureCtx.drawImage(canvas, 0, 0, 84, 84)
    const imageData = captureCtx.getImageData(0, 0, 84, 84).data
    const frame = []

    for (let y = 0; y < 84; y += 1) {
        const row = []
        for (let x = 0; x < 84; x += 1) {
            const idx = (y * 84 + x) * 4
            const r = imageData[idx]
            const g = imageData[idx + 1]
            const b = imageData[idx + 2]
            const gray = Math.round(0.299 * r + 0.587 * g + 0.114 * b)
            row.push(gray)
        }
        frame.push(row)
    }

    return frame
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
        const minY = Math.max(100, halfGap + this.height - this.images.pipe.height)
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
        if (this.bird.y + this.bird.height >= this.height || this.bird.y <= 0) {
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

            if (intersects(birdRect, topRect) || intersects(birdRect, bottomRect)) {
                return true
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
    let stepsRemaining = 0

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
    const FPS = 80
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
                const frame = captureFrame84(game.getObservationCanvas(), captureCtx)

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

            if (!actionNeeded && !waitingForAction && stepsRemaining > 0) {
                if (stepsRemaining === 4 && pendingAction === 1) {
                    game.flapQueued = true
                } else {
                    game.flapQueued = false
                }

                const done = game.step()
                stepsRemaining -= 1

                if (done) {
                    gameCount += 1
                    game.reset()
                    frameQueue.length = 0 // Clear frames on reset

                    if (predictSocket.readyState === WebSocket.OPEN) {
                        predictSocket.send(JSON.stringify({ reset: true, score: game.score }))
                    }
                    actionNeeded = true
                    waitingForAction = false
                    stepsRemaining = 0
                } else if (stepsRemaining === 0) {
                    actionNeeded = true
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
