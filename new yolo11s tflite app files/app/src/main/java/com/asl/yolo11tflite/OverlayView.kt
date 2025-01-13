package com.asl.yolo11tflite

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat

class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {

    private var results = listOf<BoundingBox>()
    private val boxPaints = mutableMapOf<Int, Paint>() // Paints for different classes
    private var textBackgroundPaint = Paint()
    private var textPaint = Paint()

    private var bounds = Rect()

    init {
        initPaints()
    }

    fun clear() {
        textPaint.reset()
        textBackgroundPaint.reset()
        boxPaints.clear()
        invalidate()
        initPaints()
    }

    private fun initPaints() {
        textBackgroundPaint.color = Color.BLACK
        textBackgroundPaint.style = Paint.Style.FILL
        textBackgroundPaint.textSize = 50f

        textPaint.color = Color.WHITE
        textPaint.style = Paint.Style.FILL
        textPaint.textSize = 50f
    }

    private fun getBoxPaintForClass(clsIndex: Int): Paint {
        return boxPaints.getOrPut(clsIndex) {
            Paint().apply {
                color = generateColorForClass(clsIndex)
                strokeWidth = 8F
                style = Paint.Style.STROKE
            }
        }
    }

    private fun generateColorForClass(clsIndex: Int): Int {
        val hue = (clsIndex * (360 / 25)) % 360 // Spread colors evenly across 26 classes
        return Color.HSVToColor(floatArrayOf(hue.toFloat(), 1f, 1f))
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)

        results.forEach {
            val left = it.x1 * width
            val top = it.y1 * height
            val right = it.x2 * width
            val bottom = it.y2 * height

            // Get paint for the class
            val boxPaint = getBoxPaintForClass(it.cls)

            // Draw bounding box
            canvas.drawRect(left, top, right, bottom, boxPaint)

            // Prepare text with class name and confidence
            val drawableText = "${it.clsName} ${"%.2f".format(it.cnf * 100)}%"

            // Measure text dimensions
            textBackgroundPaint.getTextBounds(drawableText, 0, drawableText.length, bounds)
            val textWidth = bounds.width()
            val textHeight = bounds.height()

            // Draw text background
            canvas.drawRect(
                left,
                top,
                left + textWidth + BOUNDING_RECT_TEXT_PADDING,
                top + textHeight + BOUNDING_RECT_TEXT_PADDING,
                textBackgroundPaint
            )

            // Draw text
            canvas.drawText(drawableText, left, top + bounds.height(), textPaint)
        }
    }

    fun setResults(boundingBoxes: List<BoundingBox>) {
        results = boundingBoxes
        invalidate()
    }

    companion object {
        private const val BOUNDING_RECT_TEXT_PADDING = 8
    }
}
