  *	��~jL+�@2�
YIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl::ParallelMapV2::BatchV2R�hR!@! ,��X@);����@1�/�s�J@:Preprocessing2�
yIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl::ParallelMapV2::BatchV2::Shuffle::Zip[0]::ParallelMapV2J��4*�@!�k%�E@)J��4*�@1�k%�E@:Preprocessing2�
�Iterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl::ParallelMapV2::BatchV2::Shuffle::Zip[0]::ParallelMapV2::TensorSliceI�[��?!
��[��?)I�[��?1
��[��?:Preprocessing2�
gIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl::ParallelMapV2::BatchV2::Shuffle::Zip �U+^@!{�LWGMF@)o-��x>�?1���G�^�?:Preprocessing2�
wIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl::ParallelMapV2::BatchV2::Shuffle::Zip[1]::TensorSlice�,��;�?!Ŋ��:�?)�,��;�?1Ŋ��:�?:Preprocessing2�
bIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl::ParallelMapV2::BatchV2::Shuffle�%Tp�@!�(�O�vF@)qh�.�?1�3"|��?:Preprocessing2�
PIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl::ParallelMapV2%u��?!CH{��g�?)%u��?1CH{��g�?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism���:q�?!��S��?)�uʣ�?1�L�˦�?:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch�z6�>�?!ƈ~ن�?)�z6�>�?1ƈ~ن�?:Preprocessing2x
AIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl��s]�?!��bx�?)Ü�M��?1�KU�R�?:Preprocessing2F
Iterator::Model�3�ۃ�?!�fВ�{�?)`���Y~?1O��T��?:Preprocessing2t
=Iterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCache~�ƃ-v�?!�d�g]��?)K��`?1.�QPޗ?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb��No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.