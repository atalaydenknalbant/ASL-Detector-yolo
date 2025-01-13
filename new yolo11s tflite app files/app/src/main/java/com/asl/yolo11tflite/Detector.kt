package com.asl.yolo11tflite

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStreamReader

class Detector(
    private val context: Context,
    private val modelPath: String,
    private val labelPath: String,
    private val detectorListener: DetectorListener
) {

    private var _interpreter: Interpreter? = null
    private val interpreter: Interpreter
        get() = _interpreter ?: throw IllegalStateException("Interpreter not initialized. Did you forget to call setup()?")

    private var labels = mutableListOf<String>()

    private var tensorWidth = 0
    private var tensorHeight = 0
    private var numChannel = 0
    private var numElements = 0

    private val imageProcessor = ImageProcessor.Builder()
        .add(NormalizeOp(INPUT_MEAN, INPUT_STD))
        .add(CastOp(INPUT_IMAGE_TYPE))
        .build()

    fun setup() {
//        Log.d("Detector", "Setting up with model: $modelPath.")
        val model = FileUtil.loadMappedFile(context, modelPath)
        val options = Interpreter.Options()

        val availableProcessors = Runtime.getRuntime().availableProcessors()
        options.numThreads = maxOf(1, availableProcessors - 1)

        _interpreter = Interpreter(model, options)

        val inputShape = interpreter.getInputTensor(0).shape()
        val outputShape = interpreter.getOutputTensor(0).shape()

        tensorWidth = inputShape[1]
        tensorHeight = inputShape[2]
        numChannel = outputShape[1]
        numElements = outputShape[2]

        labels = loadLabels(labelPath)
//        Log.d(
//            "Detector", "Model Input: $tensorWidth x $tensorHeight, " +
//                    "Output Channels: $numChannel, Boxes: $numElements. Labels loaded: ${labels.size}"
//        )
    }

    fun clear() {
        Log.d("Detector", "Clearing interpreter resources.")
        _interpreter?.close()
        _interpreter = null
    }

    fun detect(frame: Bitmap) {
        if (_interpreter == null) {
            Log.e("Detector", "Interpreter not initialized. Detection aborted.")
            return
        }
        if (tensorWidth == 0 || tensorHeight == 0 || numChannel == 0 || numElements == 0) {
            Log.e("Detector", "Invalid tensor shape. Detection aborted.")
            return
        }

        var inferenceTime = SystemClock.uptimeMillis()

        val resizedBitmap = Bitmap.createScaledBitmap(frame, tensorWidth, tensorHeight, false)
        val tensorImage = TensorImage(DataType.FLOAT32).apply { load(resizedBitmap) }
        val processedImage = imageProcessor.process(tensorImage)
        val imageBuffer = processedImage.buffer

        val output = TensorBuffer.createFixedSize(
            intArrayOf(1, numChannel, numElements),
            OUTPUT_IMAGE_TYPE
        )
        interpreter.run(imageBuffer, output.buffer)
        inferenceTime = SystemClock.uptimeMillis() - inferenceTime

        val bestBoxes = parseOutput(output.floatArray)
        if (bestBoxes == null) {
            detectorListener.onEmptyDetect()
            return
        }

        detectorListener.onDetect(bestBoxes, inferenceTime)
    }

    private fun loadLabels(labelFilePath: String): MutableList<String> {
        val localLabels = mutableListOf<String>()
        try {
            context.assets.open(labelFilePath).use { inputStream ->
                BufferedReader(InputStreamReader(inputStream)).use { reader ->
                    var line: String? = reader.readLine()
                    while (!line.isNullOrEmpty()) {
                        localLabels.add(line)
                        line = reader.readLine()
                    }
                }
            }
        } catch (e: IOException) {
            e.printStackTrace()
        }
        return localLabels
    }

    private fun parseOutput(array: FloatArray): List<BoundingBox>? {
        val boundingBoxes = mutableListOf<BoundingBox>()
        for (c in 0 until numElements) {
            var maxConf = -1.0f
            var maxIdx = -1

            var j = 4
            var idx = c + numElements * j
            while (j < numChannel) {
                val value = array[idx]
                if (value > maxConf) {
                    maxConf = value
                    maxIdx = j - 4
                }
                j++
                idx += numElements
            }

            if (maxConf > CONF_THRESHOLD) {
                val cx = array[c]
                val cy = array[c + numElements]
                val w = array[c + numElements * 2]
                val h = array[c + numElements * 3]

                val x1 = cx - (w / 2F)
                val y1 = cy - (h / 2F)
                val x2 = cx + (w / 2F)
                val y2 = cy + (h / 2F)

                if (x1 < 0F || x1 > 1F) continue
                if (y1 < 0F || y1 > 1F) continue
                if (x2 < 0F || x2 > 1F) continue
                if (y2 < 0F || y2 > 1F) continue

                val labelName = if (maxIdx in labels.indices) labels[maxIdx] else "Unknown"
                val confidencePercentage = maxConf * 100

//                Log.d("Detection", "Detected $labelName with ${"%.2f".format(confidencePercentage)}% confidence.")

                boundingBoxes.add(
                    BoundingBox(
                        x1 = x1, y1 = y1,
                        x2 = x2, y2 = y2,
                        cx = cx, cy = cy,
                        w = w, h = h,
                        cnf = maxConf,
                        cls = maxIdx,
                        clsName = labelName
                    )
                )
            }
        }
        if (boundingBoxes.isEmpty()) return null

        return applyNMS(boundingBoxes)
    }

    private fun applyNMS(boxes: List<BoundingBox>): MutableList<BoundingBox> {
        val sortedBoxes = boxes.sortedByDescending { it.cnf }.toMutableList()
        val selectedBoxes = mutableListOf<BoundingBox>()

        while (sortedBoxes.isNotEmpty()) {
            val first = sortedBoxes.removeAt(0)
            selectedBoxes.add(first)

            val iterator = sortedBoxes.iterator()
            while (iterator.hasNext()) {
                val nextBox = iterator.next()
                val iou = calculateIoU(first, nextBox)
                if (iou >= IOU_THRESHOLD) iterator.remove()
            }
        }
        return selectedBoxes
    }

    private fun calculateIoU(box1: BoundingBox, box2: BoundingBox): Float {
        val x1 = maxOf(box1.x1, box2.x1)
        val y1 = maxOf(box1.y1, box2.y1)
        val x2 = minOf(box1.x2, box2.x2)
        val y2 = minOf(box1.y2, box2.y2)
        val intersection = maxOf(0F, x2 - x1) * maxOf(0F, y2 - y1)
        val box1Area = box1.w * box1.h
        val box2Area = box2.w * box2.h
        return intersection / (box1Area + box2Area - intersection)
    }

    interface DetectorListener {
        fun onEmptyDetect()
        fun onDetect(boundingBoxes: List<BoundingBox>, inferenceTime: Long)
    }

    companion object {
        private const val INPUT_MEAN = 0f
        private const val INPUT_STD = 255f
        private val INPUT_IMAGE_TYPE = DataType.FLOAT32
        private val OUTPUT_IMAGE_TYPE = DataType.FLOAT32
        private const val CONF_THRESHOLD = 0.7
        private const val IOU_THRESHOLD = 0.7F
    }
}
