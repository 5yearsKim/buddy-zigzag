package gemmini

import chisel3._
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, Paths}
import java.security.MessageDigest
import org.json4s._
import org.json4s.jackson.JsonMethods.{pretty, render}
import scala.jdk.CollectionConverters._
import scala.sys.process._

/** Export resolved values, never parse Scala source or a generated C header. */
object GemminiConfigExport {

  private def obj(fields: (String, JValue)*): JObject = JObject(fields.toList)
  private def number(value: Int): JValue = JInt(BigInt(value))

  private def arithmetic(value: Data): JValue = value match {
    case f: gemmini.Float => obj(
      "kind" -> JString("float"), "bits" -> number(f.getWidth),
      "exponent_bits" -> number(f.expWidth), "significand_bits" -> number(f.sigWidth),
      "recoded" -> JBool(f.isRecoded))
    case s: SInt => obj("kind" -> JString("sint"), "bits" -> number(s.getWidth))
    case u: UInt => obj("kind" -> JString("uint"), "bits" -> number(u.getWidth))
    case other => throw new IllegalArgumentException(s"Unsupported Gemmini arithmetic: ${other.getClass.getName}")
  }

  private def git(directory: Path, arguments: String*): Option[String] = {
    val command = Seq("git", "-C", directory.toString) ++ arguments
    val output = new StringBuilder
    try {
      val status = Process(command).!(ProcessLogger(line => output.append(line).append('\n'), _ => ()))
      if (status == 0) Some(output.toString.trim) else None
    } catch { case _: java.io.IOException => None }
  }

  private def sourceRoot: Path = {
    sys.env.get("GEMMINI_SOURCE_DIR").map(Paths.get(_)).getOrElse {
      Seq(Paths.get("generators/gemmini"), Paths.get("../../generators/gemmini"), Paths.get("."))
        .find(p => Files.isDirectory(p.resolve("src/main/scala/gemmini")))
        .getOrElse(Paths.get("."))
    }.toAbsolutePath.normalize
  }

  private def scalaHash(root: Path): JValue = {
    val sources = root.resolve("src/main/scala")
    if (!Files.isDirectory(sources)) return JNull
    val stream = Files.walk(sources)
    try {
      val digest = MessageDigest.getInstance("SHA-256")
      stream.iterator.asScala.filter(p => Files.isRegularFile(p) && p.toString.endsWith(".scala"))
        .toSeq.sortBy(_.toString).foreach { path =>
          digest.update(sources.relativize(path).toString.getBytes(StandardCharsets.UTF_8))
          digest.update(0.toByte)
          digest.update(Files.readAllBytes(path))
          digest.update(0.toByte)
        }
      JString(digest.digest.map(b => f"${b & 0xff}%02x").mkString)
    } finally stream.close()
  }

  def snapshot[T <: Data : Arithmetic, U <: Data, V <: Data](
      config: GemminiArrayConfig[T, U, V], configName: String, origin: String): String = {
    require(configName.nonEmpty, "A configuration name is required for provenance")
    require(Set("elaboration", "resolved_config").contains(origin))
    val root = sourceRoot
    val chipyard = sys.env.get("CHIPYARD_SOURCE_DIR").map(Paths.get(_))
      .getOrElse(root.resolve("../..")).toAbsolutePath.normalize
    def revision(path: Path): JValue = git(path, "rev-parse", "HEAD").map(JString(_)).getOrElse(JNull)
    val dirty = git(root, "status", "--porcelain", "--untracked-files=normal", "--", "src/main/scala")
      .map(s => JBool(s.nonEmpty)).getOrElse(JNull)
    val columns = config.meshColumns * config.tileColumns
    val accRowBits = columns * config.accType.getWidth
    val data = obj(
      "schema_version" -> number(1),
      "source" -> obj(
        "config_name" -> JString(configName), "origin" -> JString(origin),
        "gemmini_revision" -> revision(root), "chipyard_revision" -> revision(chipyard),
        "gemmini_scala_dirty" -> dirty, "gemmini_scala_sha256" -> scalaHash(root)),
      "array" -> obj(
        "mesh_rows" -> number(config.meshRows), "mesh_columns" -> number(config.meshColumns),
        "tile_rows" -> number(config.tileRows), "tile_columns" -> number(config.tileColumns),
        "rows" -> number(config.meshRows * config.tileRows), "columns" -> number(columns),
        "tile_latency" -> number(config.tile_latency), "mesh_output_delay" -> number(config.mesh_output_delay)),
      "arithmetic" -> obj(
        "input" -> arithmetic(config.inputType), "weight" -> arithmetic(config.weightType),
        "accumulator" -> arithmetic(config.accType),
        "spatial_input" -> arithmetic(config.spatialArrayInputType),
        "spatial_weight" -> arithmetic(config.spatialArrayWeightType),
        "spatial_output" -> arithmetic(config.spatialArrayOutputType)),
      "scratchpad" -> obj(
        "banks" -> number(config.sp_banks), "rows_per_bank" -> number(config.sp_bank_entries),
        "row_bits" -> number(config.sp_width),
        "capacity_bits" -> JInt(BigInt(config.sp_banks) * config.sp_bank_entries * config.sp_width),
        "single_ported" -> JBool(config.sp_singleported), "read_latency" -> number(config.spad_read_delay)),
      "accumulator" -> obj(
        "banks" -> number(config.acc_banks), "rows_per_bank" -> number(config.acc_bank_entries),
        "row_bits" -> number(accRowBits),
        "capacity_bits" -> JInt(BigInt(config.acc_banks) * config.acc_bank_entries * accRowBits),
        "single_ported" -> JBool(config.acc_singleported), "read_latency" -> number(config.acc_latency),
        "sub_banks" -> number(config.acc_sub_banks)),
      "dma" -> obj("buswidth_bits" -> number(config.dma_buswidth), "max_bytes" -> number(config.dma_maxbytes)),
      "dataflow" -> JString(config.dataflow.toString),
      "external_memory" -> obj(
        "shared" -> JBool(config.use_shared_ext_mem), "tilelink" -> JBool(config.use_tl_ext_mem)))
    pretty(render(data)) + "\n"
  }


}

/** A fast resolved-object snapshot; does not elaborate or build a Chipyard SoC. */
object GemminiConfigExportMain {
  def main(args: Array[String]): Unit = {
    require(args.length == 2 && args(0) == "--output",
      "Usage: gemmini.GemminiConfigExportMain --output /path/to/parameters.json")
    val output = Paths.get(args(1)).toAbsolutePath.normalize
    Files.createDirectories(output.getParent)
    val text = GemminiConfigExport.snapshot(
      GemminiFPConfigs.FP32DefaultConfig, "gemmini.GemminiFPConfigs.FP32DefaultConfig", "resolved_config")
    Files.write(output, text.getBytes(StandardCharsets.UTF_8))
    println(s"Exported resolved FP32 configuration to $output (no SoC elaboration)")
  }
}
