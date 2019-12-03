package com.stripe.agate
package tensor

import cats.effect.{IO, Resource}
import java.io.{File, OutputStream, RandomAccessFile}
import java.nio.{ByteBuffer, ByteOrder, FloatBuffer, MappedByteBuffer}
import java.nio.channels.FileChannel
import java.nio.file.Path
import java.util.Arrays

import Shape.Dim
import Util.RequireIntOps

sealed trait Storage[A] {
  def apply(idx: Long): A
  def slice(skipOff: Long): Storage[A]
  def writeIntoStream(os: OutputStream, dim: Dim, len: Long)(implicit tb: ToBytes[A]): Unit
  def writeInto(out: WritableStorage[A], targetOffset: Long, dim: Dim, len: Long): Unit
}

object Storage {
  final case class ArrayStorage[@specialized A](data: Array[A], offset: Int) extends Storage[A] {
    @inline final def apply(idx: Long): A =
      data(offset + idx.toInt)

    final def slice(skipOff: Long): ArrayStorage[A] =
      ArrayStorage(data, (offset + skipOff).requireInt)

    final def writeInto(out: WritableStorage[A], targetOffset: Long, dim: Dim, len: Long): Unit =
      if (len == 0L) ()
      else {
        val Shape.Dim(off, stride) = dim
        out.writeFrom(
          data,
          offset + off.toInt,
          targetOffset.requireInt,
          stride.requireInt,
          len.requireInt
        )
      }

    final def writeIntoStream(os: OutputStream, dim: Dim, len: Long)(
        implicit tb: ToBytes[A]
    ): Unit =
      if (len == 0L) ()
      else {
        val Dim(off, stride) = dim
        writeIntoStreamImpl(os, off.requireInt, stride.requireInt, len.requireInt, tb)
      }

    private def writeIntoStreamImpl(
        os: OutputStream,
        off: Int,
        stride: Int,
        len: Int,
        tb: ToBytes[A]
    ): Unit = {
      var i = 0
      var j = offset + off
      while (i < len) {
        tb.put(os, data(j))
        i += 1
        j += stride
      }
    }
  }

  final case class FloatBufferStorage(buffer: FloatBuffer, offset: Int) extends Storage[Float] {
    def apply(idx: Long): Float =
      buffer.get(idx.toInt + offset)

    def slice(skipOff: Long): Storage[Float] =
      FloatBufferStorage(buffer, (offset + skipOff).requireInt)

    final def writeIntoStream(os: OutputStream, dim: Dim, len: Long)(
        implicit tb: ToBytes[Float]
    ): Unit = {
      val Dim(off, stride) = dim
      var i = 0
      var j = (offset + off).requireInt
      val strideI: Int = stride.requireInt
      val limit: Int = len.requireInt
      while (i < limit) {
        tb.put(os, buffer.get(j))
        i += 1
        j += strideI
      }
    }

    def writeInto(out: WritableStorage[Float], off: Long, dim: Dim, len: Long): Unit =
      out match {
        case WritableStorage.ArrayStorage(output, off0) =>
          dim match {
            case Dim(offDim, 1L) =>
              val dupBuffer = buffer.duplicate
              dupBuffer.position((offset + offDim).requireInt)
              val ignore = dupBuffer.get(output, off.toInt + off0, len.requireInt)
            case Dim(offDim, strideDim) =>
              // slow case, because we have non unital stride
              var i = 0
              var j = (offset + offDim).requireInt
              val strideI = strideDim.requireInt
              while (i < len) {
                output(off0 + i) = buffer.get(j)
                i += 1
                j += strideI
              }
          }
      }

    def copyToArrayStorage: Storage.ArrayStorage[Float] = {
      val len = (buffer.limit() - offset).toLong
      val writable = StorageAllocator[Float].allocate(len)
      writeInto(writable, 0L, Shape.SingleDim, len)
      writable.toArrayStorage
    }

    def size: Long =
      buffer.limit() - offset.toLong
  }

  final case class Chunked[@specialized A](
      chunks: Array[Storage[A]],
      maxChunkSize: Long,
      totalOffset: Long
  ) extends Storage[A] {
    override def toString =
      "Chunked(" + chunks.mkString("Array(", ", ", ")") + s", $maxChunkSize, $totalOffset)"

    private def chunkOf(rawIdx: Long): Int =
      (rawIdx / maxChunkSize).toInt

    def apply(idx: Long): A = {
      val rawIdx = idx + totalOffset
      val chunk = chunkOf(rawIdx)
      val off = chunk * maxChunkSize
      chunks(chunk)(rawIdx - off)
    }

    def slice(skipOff: Long): Storage[A] = copy(totalOffset = totalOffset + skipOff)

    private def adjustLen(currentChunk: Int, off: Long, stride: Long, len: Long): Long =
      if (currentChunk == (chunks.length - 1) || stride == 0L) {
        // this is the last chunk:
        len
      } else {
        // we can know the size of this:
        val items = ((currentChunk + 1) * maxChunkSize) - off
        (items + stride - 1) / stride // ceil(items / stride)
      }

    private def foldLen[B](dim: Dim, len: Long, init: B)(fn: (Storage[A], B, Dim, Long) => B): B = {
      val stride = dim.stride

      // start at a given offset, write len bytes
      def writeChunk(currentChunk: Int, off: Long, init: B, len: Long): B = {
        val chunkLen = adjustLen(currentChunk, off, stride, len)
        val chunkOffset = off - currentChunk * maxChunkSize
        val chunkDim = Dim(chunkOffset, stride)
        val nextLen = len - chunkLen
        if (nextLen <= 0) {
          fn(chunks(currentChunk), init, chunkDim, len)
        } else {
          val next = fn(chunks(currentChunk), init, chunkDim, chunkLen)
          writeChunk(currentChunk + 1, off + chunkLen * stride, next, nextLen)
        }
      }
      val sourceOffset = totalOffset + dim.offset
      val currentChunk = chunkOf(sourceOffset)
      if (len > 0) writeChunk(currentChunk, sourceOffset, init, len)
      else init
    }

    def writeIntoStream(os: OutputStream, dim: Dim, len: Long)(implicit tb: ToBytes[A]): Unit =
      foldLen(dim, len, ()) { (storage, _, d, l) =>
        storage.writeIntoStream(os, d, l)(tb)
      }

    def writeInto(out: WritableStorage[A], offset: Long, dim: Dim, len: Long): Unit = {
      // the offset here is the target offset, not the source offset
      foldLen(dim, len, offset) { (s, o, d, l) =>
        s.writeInto(out, o, d, l)
        o + l
      }
      ()
    }
  }

  def storageFromBuffer(dt: DataType, b: ByteBuffer, offset: Int): Storage[dt.Elem] =
    dt match {
      case f: DataType.Float32.type =>
        DataType
          .maybeElemIs(f, dt)
          .get
          .substitute[Storage](
            FloatBufferStorage(b.asFloatBuffer, offset)
          )
      case _ =>
        sys.error(s"unsupported datatype: $dt")
    }

  private def openRandomAccessFileChannel(file: File): Resource[IO, (Long, FileChannel)] =
    Resource
      .make(IO.delay(new RandomAccessFile(file, "r"))) { file =>
        IO.delay(file.close())
      }
      .flatMap { file =>
        Resource.make(IO.delay((file.length, file.getChannel))) {
          case (_, ch) => IO.delay(ch.close())
        }
      }

  /**
   * Load storage off disk, given a count of A (which is to say, count is not in bytes)
   */
  def loadMapped(
      dt: DataType,
      path: Path,
      count: Long,
      maxChunkSizeInBytes: Int
  ): Resource[IO, Storage[dt.Elem]] = {
    val tb = ToBytes.forDataType(dt)
    tb.strategy match {
      case _: ToBytes.Strategy.VarLength[_] =>
        Resource.liftF(
          IO.raiseError(
            new IllegalArgumentException(s"unsupported datatype: $dt")
          )
        )
      case ToBytes.Strategy.FixedLength(step) =>
        if (maxChunkSizeInBytes < step) {
          Resource.liftF(
            IO.raiseError(
              new IllegalArgumentException(s"maxChunkSizeInBytes=${maxChunkSizeInBytes} < $step")
            )
          )
        } else {
          openRandomAccessFileChannel(path.toFile).map {
            case (length, channel) =>
              val size = count * step
              if (length < size) {
                throw TensorException.InsufficentBytesToLoad(length, dt, size)
              }
              val maxSingleBytes = (maxChunkSizeInBytes / step) * step
              @annotation.tailrec
              def loop(
                  off: Long,
                  count: Long,
                  acc: List[(Storage[dt.Elem], Long)]
              ): List[(Storage[dt.Elem], Long)] =
                if (count <= 0) acc.reverse
                else {
                  val sizeBytes = count * step
                  // definitely an integer multiple of step
                  val thisSize = sizeBytes min maxSingleBytes
                  val offBytes = off * step
                  val mapped = channel.map(FileChannel.MapMode.READ_ONLY, offBytes, thisSize)
                  mapped.order(ByteOrder.LITTLE_ENDIAN)
                  val store = storageFromBuffer(dt, mapped, 0)
                  val thisCount = thisSize / step
                  val nextCount = count - thisCount
                  val nextOff = off + thisCount
                  loop(nextOff, nextCount, (store, off) :: acc)
                }

              val (stores, offs) = loop(0L, count, Nil).unzip
              stores match {
                case single :: Nil => single
                case many =>
                  Chunked(many.toArray, maxSingleBytes / step, 0L)
              }
          }
        }
    }
  }
}
