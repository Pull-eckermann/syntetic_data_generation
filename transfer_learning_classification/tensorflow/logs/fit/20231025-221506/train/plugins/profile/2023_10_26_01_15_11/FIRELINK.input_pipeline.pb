  *	 �rh��@2�
yIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl::ParallelMapV2::BatchV2::Shuffle::Zip[0]::ParallelMapV2 t$����$@!�3Hkm�H@)t$����$@1�3Hkm�H@:Preprocessing2�
YIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl::ParallelMapV2::BatchV2�E����4@!��[���X@)��y'$@1]!���G@:Preprocessing2�
gIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl::ParallelMapV2::BatchV2::Shuffle::Zip 	S�K�G%@!ʋ�VI@)ձJ�^�?1fH�W���?:Preprocessing2�
�Iterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl::ParallelMapV2::BatchV2::Shuffle::Zip[0]::ParallelMapV2::TensorSlice �yq��?!,�	/c��?)�yq��?1,�	/c��?:Preprocessing2�
bIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl::ParallelMapV2::BatchV2::Shuffle �an�r%@!�N�g.�I@)-�Yf��?1���B���?:Preprocessing2�
wIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl::ParallelMapV2::BatchV2::Shuffle::Zip[1]::TensorSlice �X�ׯ?!�r�4K��?)�X�ׯ?1�r�4K��?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism[�T�2�?!�	6uK��?)yY|E�?1t2[����?:Preprocessing2�
PIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl::ParallelMapV2��%���?!k,WR:�?)��%���?1k,WR:�?:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch|,G�@�?!�����?)|,G�@�?1�����?:Preprocessing2x
AIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl�'���?!R	�!�?)>���d��?1t���Ý�?:Preprocessing2F
Iterator::Model<�8b->�?!i�\SK�?)O!W�Yp?1��4q<x�?:Preprocessing2t
=Iterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCache�u�+.��?!H�'��?)/��$�\?1^��x�9�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb��No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.