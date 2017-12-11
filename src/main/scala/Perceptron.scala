class Perceptron(val trainingRate: Double = 0.2) {

  import scala.util.Random

  private val minVal = -100.0
  private val maxVal = 100.0

  var weights: Array[Double] = Array(0.0, 0.0, 0.0)

  def nextFloat = {
    minVal + (maxVal - minVal) * Random.nextFloat()
  }

  def fitFuction(input: Double) = if (input >= 0) 1 else -1

  def activateFunction(inputs: Array[Double]) = {
    val sum = weights
      .zip(inputs)
      .foldLeft[Double](0.0)((curSum, curTup) => curSum + curTup._1 * curTup._2)
    fitFuction(sum)
  }

  def train(inputs: Array[Double], expectedOutput: Int) = {
    require(inputs.length == weights.length)
    val dw = expectedOutput - activateFunction(inputs)
    for (i <- 0 until weights.length) {
      weights(i) = weights(i) + dw * inputs(i) * trainingRate
    }

  }

}

class Trainer(private val x: Double, private val y: Double, private val f: Double => Double) {
  val inputs = Array(x, y, 1.0)
  val output = if (y >= f(x)) 1 else -1
}

object main extends App {
  val p = new Perceptron(0.1)
  val n = 1000
  val f = (x:Double) => 3 * x + 2


  val trainingList = for (i <- 0 until n) yield new Trainer(p.nextFloat, p.nextFloat, f)

  for ((point, idx) <- trainingList.zipWithIndex) {
    p.train(point.inputs, point.output)

    val correctedGuest = trainingList.filter(_point => _point.output == p.activateFunction(_point.inputs)).size
    println(s"$idx : $correctedGuest")

  }
}
