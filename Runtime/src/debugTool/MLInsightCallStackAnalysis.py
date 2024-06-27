'''
@author: Steven (Jiaxun) Tang <jtang@umass.edu>
@author: Tongping Liu <tongping.liu@bytedance.com>
'''

from collections import defaultdict
import numpy as np
import re
from matplotlib import pyplot as plt
import networkx as nx
from collections import defaultdict

'''
Data structures
'''


class CallStackInfo:
    def __init__(self, count, callStackLines, allocationSize):
        '''
        A class that records callstack-related information
        :param count: How many pytorch memory allocations have this callstack, Or this callstack has appeared how many times.
        :param callStackLines: A list representing a callstack，Each list element corresponds to one line of the printed callstack line.
        :param allocationSize: The size of allocation in bytes.
        '''
        self.count = count
        # Whether this callstack is redundant, if true it means the callstack is not redundant.
        self.notRedundant = True
        self.lines = callStackLines
        self.allocationSize = allocationSize


'''
CallStack related information
'''
# callstackInfoList[i] points to a corresponding "class CallStackInfo". The index of this array is "callstackId".
callstackInfoList = []
# Map a python callstack to an id
callstackIdMap = dict()
# The index of this array is the id of a callstack line, and the value of this array is the string of callstack line. The purpose is to store numbers in callstack array rather than string to improve performance.
callStackLineIdStringMap = []
# The key of this dictionary is the string of callstack line, the value of this dictionary is callstack line id. Users can use this id to get the callstack line string with "callStackLineIdStringMap".
callStackLineStringIdMap = dict()

'''
Sequence related information
'''
# The index of this array is called "sequenceId", which indicates to the order of callstack appearance on timeline, and the array element is callstack id.
callStackSequence = []

'''
Link allocation with deallocation
'''
# Used for friendly output. The index of this array is the sequenceId of the allocation. The value of this array is a list of callStackIds that has been freed right after the sequence.
freeSequence = []

# The index of this array is the the allocation sequenceId. The value of this array is [the allocation sequenceId, the deallocaiton sequenceId].
# The purpose of storing an extra allocation sequenceId in array element is to provide sequenceId after removing redundant callstacks.
allocationDeallocationSequnceIdPairs = []

# Helper variables
unmatchedAllocationAddressDict = dict()  # A set storing currently unmatched allocation address
lastAllocationAddress = None

# This list is used to store allocations and deallations of interest in the form of [allocaiton sequenceId, deallocation sequenceId]. In the end, these allocaiton pairs will be shown on the plot.
allocationDeallocationPairsToPlot = []  # A pair of allocation sequence indicating the allocation or free of an address. Used to plot individual deallocations

'''
Boundary related information. 
It is difficult to perform analysis on the entire log, so I printed some extra logs that can split the output into smaller sections related to each transformer blocks, steps .etc.
'''
pytorchAllocationMarker = re.compile(r'trackPytorchAllocation.ptr.*')
pytorchDeAllocationMarker = re.compile(r'raw_delete_proxy.ptr.*')
# The following regex expressions helps to match the boundary output of the log file
pyCallstackPrintStartMarker = re.compile(r'=*HYBRIDSTACK.LEVELS.START')
pyCallstackPrintEndMaker = re.compile(r'=*HYBRIDSTACK.LEVELS.END')
# preparationFinishedMarker = re.compile(r'\[Rank.\d+\].Finished.steps.before.training')
# finishedStepForwardMarker = re.compile(r'\[Rank.\d+\].step.\d+.finished.the.forwarding')
# finishedStepBackwardMarker = re.compile(r'\[Rank.\d+\].step.\d+.finished.the.backward')
# finishedStepOptimizerMarker = re.compile(r'\[Rank.\d+\].step.\d+.finished.the.optimizer')
# allocationMarker = re.compile(r'AllocationSize:.\d+')
# startedLayerForwardMarker = re.compile(r'\[Rank.\d+\].layer.forward.started.name:model.decoder.layers.\d+')
# finishedLayerForwardMarker = re.compile(r'\[Rank.\d+\].layer.forward.finished.name:model.decoder.layers.\d+')
# startedLayerBackwardMarker = re.compile(r'\[Rank.\d+\].layer.backward.started.name:model.decoder.layers.\d+')
# finishedLayerBackwardMarker = re.compile(r'\[Rank.\d+\].layer.backward.finished.name:model.decoder.layers.\d+')

# Used to plot the boundary information. The element of this list is in the form of: [sequenceId,friendlyName]. sequenceId is the sequenceId of the last memory allocations before boundary log print. friendlyName is the name shown in the final plot.
callSequenceBoundaryPlotData = []

# The value of this array is [start sequenceId, end sequenceId]. Each value corresponds to a region marked by two boundaries.
callStackSequencePairs = []

'''
Allocaiton size related information
'''
# A list that records all allocation size found in the log. It is mainly used to analyze the distribution of the allocationSize.
allocationSizeList = []

'''
Log parsing related global variable.
'''
lastSequenceId = -1
lastCallStackId = -1


def callStackHasString(targetString, callStackInfo):
    '''
    A utility function used to check whether the collected callstack has certain string or not
    :param targetString: A string to match
    :param callStackInfo: The string should be searched in this callStackInfo
    :return: Whether targetString is inside the callstack
    '''
    hasString = False
    for _lineId in callStackInfo.lines:
        # If a callstack contains "Line:", then this callstack is not empty
        if targetString in callStackLineIdStringMap[_lineId]:
            hasString = True
            break
    return hasString


'''
Log parser
'''


def handlePytrochCallStack(curLineId):
    global firstLineId, curCallStackId, sequenceId
    # Read the callstack content
    currentCallStackBuffer = []
    curLineId += 1
    while not pyCallstackPrintEndMaker.match(lines[curLineId]):
        # Read until the end of the callstack
        firstLineId = -1
        # Strip level and "Function:" keyword and put into the vocabulary
        functionStrIndex = lines[curLineId].find('Function: ')
        lineStrip = lines[curLineId]

        if functionStrIndex > 0:
            lineStrip = lineStrip[functionStrIndex + 10:]
        if lineStrip not in callStackLineStringIdMap:
            # If not found, allocate lineId
            firstLineId = len(callStackLineIdStringMap)
            assert (len(callStackLineIdStringMap) == len(callStackLineStringIdMap))
            callStackLineStringIdMap[lineStrip] = firstLineId
            callStackLineIdStringMap.append(lineStrip)
        else:
            # If found, get lineId
            firstLineId = callStackLineStringIdMap[lineStrip]
        currentCallStackBuffer.append(firstLineId)
        curLineId += 1
    curCallStackId = None
    if tuple(currentCallStackBuffer) in callstackIdMap:
        curCallStackId = callstackIdMap[tuple(currentCallStackBuffer)]
        curCallStackCounter = callstackInfoList[curCallStackId]
        curCallStackCounter.count += 1
    else:
        curCallStackId = len(callstackIdMap)
        assert (len(callstackIdMap) == len(callstackInfoList))
        callstackIdMap[tuple(currentCallStackBuffer)] = curCallStackId
        curCallStackInfo = CallStackInfo(1, currentCallStackBuffer, 0)
        callstackInfoList.append(curCallStackInfo)
    callStackSequence.append(curCallStackId)
    sequenceId = len(callStackSequence) - 1
    # print('Memory allocation stack_id=%d sequence_id=%d' %(stackIdCounter,len(callStackSequence)-1))
    print('CallStackId:', curCallStackId, 'SequenceId:', sequenceId)
    unmatchedAllocationAddressDict[lastAllocationAddress] = (sequenceId, curCallStackId)
    return curLineId


def handleBoundaryMarker(friendlyName):
    global callStackSequence, callstackInfoList, lastCallStackId, lastSequenceId
    sequenceId = len(callStackSequence) - 1
    callStackId = len(callstackInfoList) - 1
    print('StackIdCoutner=(%d,%d] SequenceCounter=(%d,%d]' % (lastCallStackId, callStackId,
                                                              lastSequenceId, sequenceId))
    allocationSequence = callStackSequence[lastSequenceId + 1:sequenceId + 1]
    if len(allocationSequence) > 0:
        # Save this log for later summary prints
        callStackSequencePairs.append([lastSequenceId, sequenceId])
        callStackSequencePairs.append(lines[curLineId])
        # Print sequence amid other logs
        print('[%d,%d] 长度 %d' % (lastSequenceId + 1, sequenceId, len(allocationSequence)), allocationSequence)
        # print('Deallocaiton Sequence',freeSequence[lastSequenceId + 1:sequenceId + 1])
    lastCallStackId = callStackId
    lastSequenceId = sequenceId
    print(lines[curLineId])


with open('/tmp/mlinsight_python_32485_8683316771944489/python_32550/log.bkp', 'r') as f:
    # Convert file content into a list
    lines = [line.strip('\n') for line in f.readlines()]
    # i records the current file reading location
    curLineId = 0
    while True:
        if curLineId == len(lines):
            # The file has been parsed
            break
        if pyCallstackPrintStartMarker.match(lines[curLineId]):
            # Handle the pytorch callstack. Pytorch callstack may occupy multiple lines, so we need to modify file curLineId as well
            curLineId = handlePytrochCallStack(curLineId)
        else:
            # No match, simply print out the log
            print(lines[curLineId])
        curLineId += 1

'''
Visualize the allocation sequence
'''

# Three variables used to calculate the memory allocation size of different type of objects
validSize = 0
initSize = 0
emptySize = 0
gradPartitionSize = 0
# initSizeList is used to visualize the relationship between allocation count and allocation size for __init__ variables. 
initSizeList = []

for callstackId in range(len(callstackInfoList)):
    # For every callstack
    callStackInfo = callstackInfoList[callstackId]
    # Find the first non allocation lines
    firstLineId = None
    for _lineId in callStackInfo.lines:
        # If a callstack contains "Line:", then this callstack is not empty
        if 'Line:' in callStackLineIdStringMap[_lineId]:
            firstLineId = _lineId
            break
    if firstLineId is None:
        # Callstack is empty. Currently we do not mark empty callstacks as redundant.
        callStackInfo.notRedundant = True
        emptySize += callStackInfo.allocationSize * callStackInfo.count
        print('Callstack %d is empty' % (callstackId))
    elif '__init__' in callStackLineIdStringMap[firstLineId]:
        # Callstack is invalid because it starts without __init__.
        callStackInfo.notRedundant = False
        initSize += callStackInfo.allocationSize * callStackInfo.count
        initSizeList.append(
            [callstackId, callStackInfo.count, callStackInfo.count * callStackInfo.allocationSize / 1024 / 1024])
    elif callStackHasString('__reduce_and_partition_ipg_grads', callStackInfo):
        # Callstack is invalid because it contains __reduce_and_partition_ipg_grads.
        callStackInfo.notRedundant = False
        gradPartitionSize += callStackInfo.allocationSize * callStackInfo.count
    else:
        # Callstack is valid because it does not has __init__ and empty
        validSize += callStackInfo.allocationSize * callStackInfo.count

print('validSize', validSize, 'initSize', initSize, 'emptySize', emptySize, 'gradPartitionSize', gradPartitionSize)

'''
First return value:rltIndexSequence A list of indexes of the non-redundant callstacks. The list element indicates which callstack should be not removed.
Second return value:rltSequenceIdMap Map index that is not removed before to the removed index. The key is the id before removal, the value is the id after removal.
'''


def removeRedundantCallStacks(callStackSequence):
    rltSequence = []
    rltIndexSequence = []
    rltIdMap = []  # Map the original sequence id to the new sequence id that does not include removed elements. If the element is -1, then this element indicates a removed sequence
    deletedObjNumber = 0
    for index, callStackId in enumerate(callStackSequence):
        if callstackInfoList[callStackId].notRedundant:
            rltIndexSequence.append(index)
            rltSequence.append(callStackId)
            rltIdMap.append(index - deletedObjNumber)
        else:
            # append placeholder sequenceId and make it possible to still get the correct sequence id by index
            deletedObjNumber += 1
            rltIdMap.append(index - deletedObjNumber)

    return rltIndexSequence, rltIdMap


def onlyKeepRedundantCallStacks(callStackSequence):
    rltSequence = []
    rltIndexSequence = []
    for index, callStackId in enumerate(callStackSequence):
        if not callstackInfoList[callStackId].notRedundant:
            rltIndexSequence.append(index)
            rltSequence.append(callStackId)
    return rltIndexSequence


# redundantCallStackSequenceIndex=onlyKeepRedundantCallStacks(callStackSequence)
# print(callSequenceBoundary)
# exit(0)
print('===========Print callstack dedup', len(callstackInfoList))
for curCallStackId in range(len(callstackInfoList)):
    callStackInfo = callstackInfoList[curCallStackId]

    callStackLines = []
    callStackLines.append('The following callstack id=%d with levels=%d occurred %d times valid=%s' % (
        curCallStackId, len(callStackInfo.lines), callStackInfo.count, str(callStackInfo.notRedundant)))
    print(callStackLines[-1])
    for firstLineId in callStackInfo.lines:
        callStackLines.append(callStackLineIdStringMap[firstLineId])
        print(callStackLines[-1])
    print()
    with open('/tmp/' + str(curCallStackId) + '.txt', 'w') as f:
        f.writelines([line + '\n' for line in callStackLines])

exit(0)
print('===========Measure callstack similarity')


def calculateCallstackSimilarity(stackA, stackB):
    # Find common lines
    return len(set(stackA) & set(stackB))


# Plt initSize distribution
# fig,(ax1,ax2)=plt.subplots(2,1,sharex=True)
# ax1.bar([i[0] for i in initSizeList],[i[1] for i in initSizeList])
# # ax1.set_xlabel('CallStack Id (Only shows __init__)')
# ax1.set_ylabel('Allocation Count')
# ax2.bar([i[0] for i in initSizeList],[i[2] for i in initSizeList])
# ax2.set_xlabel('CallStack Id (Only shows __init__)')
# ax2.set_ylabel('Allocation Size (MB)')
# plt.show()

# Measure similarity
# similarityList = []
# similarityMatrix = np.zeros(shape=(len(callStackVocabulary), len(callStackVocabulary)), dtype=float)
# for i in range(len(callStackVocabulary)):
#     for j in range(i + 1, len(callStackVocabulary)):
#         callStackI = callStackVocabulary[i]
#         callStackJ = callStackVocabulary[j]
#         similarity = calculateCallstackSimilarity(callStackI, callStackJ) / max(len(callStackI), len(callStackJ))
#         similarityMatrix[i, j] = similarity
#         similarityMatrix[j, i] = similarity
#         similarityList.append(similarity)
#
# similarityList = np.array(similarityList)
# count, bins_count = np.histogram(similarityList, bins=20)
# # finding the PDF of the histogram using count values
# pdf = count / sum(count)
# # using numpy np.cumsum to calculate the CDF
# # We can also find using the PDF values by looping and adding
# cdf = np.cumsum(pdf)
# plt.scatter(bins_count[1:], pdf, color="red", label="PMF")
# plt.plot(bins_count[1:], cdf, label="CDF")
# plt.title('Callstack similarity distribution')
# plt.legend()
# plt.show()

# for i in range(len(callstackCounterMap)):
#     print('')
# plt.figure()
# plt.matshow(similarityMatrix > 0.75)
# plt.title('Callstack similarity')
# plt.xlabel('Call Stack id1')
# plt.ylabel('Call Stack id2')
# plt.colorbar()
# plt.show()

print('===========Callstack sequence')
for callStackId in callStackSequence:
    print(callStackId, end='\t')
print()
# print(callStackSequence)
# print(freeSequence)
for callStackId in freeSequence:
    print(callStackId, end='\t')
print()

# fig, ax = plt.subplots()
# plt.scatter(np.arange(len(redundantCallStackSequence)), np.array(redundantCallStackSequence), s=2)
# trans = ax.get_xaxis_transform()
# plt.ylabel('Callstack ID')
# plt.xlabel('Callstack Sequence (Redundant Only)')
# plt.show()
#


fig, ax = plt.subplots()
plt.scatter(np.arange(len(callStackSequence)), np.array(callStackSequence), s=2)
trans = ax.get_xaxis_transform()
for boundary in callSequenceBoundaryPlotData:
    yAxis = 1.0
    textColor = 'black'
    if boundary[1].endswith('_S'):
        yAxis = 0.8
        textColor = 'grey'
    plt.text(boundary[0], yAxis, boundary[1], rotation='vertical', transform=trans, color=textColor)
    plt.axvline(boundary[0])
plt.ylabel('Callstack ID')
plt.xlabel('Callstack Sequence')

# Plot specified deallocation pairs
deallocationSequenceId = [7426, 7418, 7426, 7424, 7424, 7420, 7416, 7408, 7407, 7401, 7407, 7407, 7394, 7029, 7028,
                          7022, 7028, 7028, 6993, 6985, 6993, 6991, 6991, 6987, 6983, 6975, 6974, 6968, 6974, 6974,
                          6961, 6953, 6952, 6946, 6952, 6952, 6917, 6909, 6917, 6915, 6915, 6911, 6907, 6899, 6898,
                          6892, 6898, 6898, 6885, 6877, 6876, 6870, 6876, 6876, 6841, 6833, 6841, 6839, 6839, 6835,
                          6831, 6823, 6822, 6816, 6822, 6822, 6809, 6801, 6800, 6794, 6800, 6800, 6765, 6757, 6765,
                          6763, 6763, 6759, 6755, 6747, 6746, 6393, 6746, 6746, 6386, 6378, 6377, 6371, 6377, 6377,
                          6342, 6334, 6342, 6340, 6340, 6336, 6332, 6324, 6323, 6317, 6323, 6323, 6310, 6302, 6301,
                          6295, 6301, 6301, 6266, 6258, 6266, 6264, 6264, 6260, 6256, 6248, 6247, 6241, 6247, 6247,
                          6234, 6226, 6225, 6219, 6225, 6225, 6190, 6182, 6190, 6188, 6188, 6184, 6180, 6172, 6171,
                          6165, 6171, 6171, 6158, 6150, 6149, 6143, 6149, 6149, 6114, 6106, 6114, 6112, 6112, 6108,
                          5747, 5739, 5738, 5732, 5738, 5738, 5725, 5717, 5716, 5710, 5716, 5716, 5681, 5673, 5681,
                          5679, 5679, 5675, 5671, 5663, 5662, 5656, 5662, 5662, 5649, 5641, 5640, 5634, 5640, 5640,
                          5605, 5597, 5605, 5603, 5603, 5599, 5595, 5587, 5586, 5580, 5586, 5586, 5573, 5565, 5564,
                          5558, 5564, 5564, 5529, 5521, 5529, 5527, 5527, 5523, 5519, 5511, 5510, 5504, 5510, 5510,
                          5497, 5489, 5488, 5482, 5488, 5488, 5096, 5088, 5096, 5094, 5094, 5090, 5086, 5078, 5077,
                          5071, 5077, 5077, 5064, 5056, 5055, 5049, 5055, 5055, 5020, 5012, 5020, 5018, 5018, 5014,
                          5010, 5002, 5001, 4995, 5001, 5001, 4988, 4980, 4979, 4973, 4979, 4979, 4944, 4936, 4944,
                          4942, 4942, 4938, 4934, 4926, 4925, 4919, 4925, 4925, 4912, 4904, 4903, 4897, 4903, 4903,
                          4868, 4860, 4868, 4866, 4866, 4862, 4858, 4850, 4849, 4843, 4849, 4849, 4836, 4828, 4827,
                          4821, 4827, 4827, 4435, 4427, 4435, 4433, 4433, 4429, 4425, 4417, 4416, 4410, 4416, 4416,
                          4403, 4395, 4394, 4388, 4394, 4394, 4359, 4351, 4359, 4357, 4357, 4353, 4349, 4341, 4340,
                          4334, 4340, 4340, 4327, 4319, 4318, 4312, 4318, 4318, 4283, 4275, 4283, 4281, 4281, 4277,
                          4273, 4265, 4264, 4258, 4264, 4264, 4251, 4243, 4242, 4236, 4242, 4242, 4207, 4199, 4207,
                          4205, 4205, 4201, 4197, 4189, 4188, 4182, 4188, 4188, 4175, 4167, 4166, 3796, 4166, 4166,
                          3767, 3759, 3767, 3765, 3765, 3761, 3757, 3749, 3748, 3742, 3748, 3748, 3735, 3727, 3726,
                          3720, 3726, 3726, 3691, 3683, 3691, 3689, 3689, 3685, 3681, 3673, 3672, 3666, 3672, 3672,
                          3659, 3651, 3650, 3644, 3650, 3650, 3615, 3607, 3615, 3613, 3613, 3609, 3605, 3597, 3596,
                          3590, 3596, 3596, 3583, 3575, 3574, 3568, 3574, 3574, 3539, 3531, 3539, 3537, 3537, 3533,
                          3529, 3521, 3520, 3514, 3520, 3520, 3507, 3499, 3498, 3493, 3498, 3498]
for curSequence in deallocationSequenceId:
    plt.axvline(curSequence, color='green')

# Plot allocation pairs
# for curPair in allocationPairsToPlot:
#     print('Allocation pair', curPair)
#     plt.axvline(curPair[0], color='red')
#     plt.text(curPair[0], .5, curPair[2], rotation='vertical', transform=trans)
#     plt.axvline(curPair[1], color='green')
#     plt.text(curPair[1], .5, curPair[2], rotation='vertical', transform=trans)
plt.show()

# Test whether all callstacks are valid, remove invalid ones such as
'''
Insert deallocation in allocation pairs
'''
for elem in callStackSequencePairs:
    if type(elem) is str:
        print(elem)
        continue
    assert (type(elem) is list)
    (lastSequenceId, sequenceId) = elem
    aList = callStackSequence[lastSequenceId + 1:sequenceId + 1]
    removeRedundantIndex, _ = removeRedundantCallStacks(
        aList)  # Callstack allocation list with invalid callstacks removed.
    aList_redundantOnlyIndex = onlyKeepRedundantCallStacks(
        aList)  # Callstack allocation list with invalid callstacks removed.

    fList = freeSequence[lastSequenceId + 1:sequenceId + 1]
    print('Allocation sequence [%d,%d] 长度 %d' % (lastSequenceId + 1, sequenceId, len(aList)), aList)
    # print('Allocation sequence remove redundant [%d,%d] 长度 %d' % (lastSequenceId + 1, sequenceId, len(removeRedundantIndex)), [aList[i] for i in removeRedundantIndex])
    # print('Allocation sequence redundant only [%d,%d] 长度 %d' % (lastSequenceId + 1, sequenceId, len(aList_redundantOnlyIndex)), [aList[i] for i in aList_redundantOnlyIndex])

    outputDList = []
    outputDList_removeRedundant = []
    for curLineId in range(len(aList)):
        outputDList.append(str(aList[curLineId]))
        outputDList.extend(['-' + str(abs(j)) for j in fList[curLineId]])

        if callstackInfoList[aList[curLineId]].notRedundant:
            outputDList_removeRedundant.append(str(aList[curLineId]))
        for j in fList[curLineId]:
            if callstackInfoList[abs(j)].notRedundant:
                outputDList_removeRedundant.append('-' + str(abs(j)))

    # print('Allocation sequence with de-allocations [%d,%d]  长度 %d' % (lastSequenceId + 1, sequenceId,len(outputDList)),str(outputDList).replace("'", ''))
    # print('Allocation sequence with de-allocations remove redundant [%d,%d] 长度 %d' % (lastSequenceId + 1, sequenceId,len(outputDList_removeRedundant)),str(outputDList_removeRedundant).replace("'", ''))

# print('===========Allocation size distribution')
# plt.figure()
# count, bins_count = np.histogram(np.array(allocationSizeList), bins=2000)
# plt.ylabel('Count')
# plt.xlabel('Allocation Size')
# plt.title('Histogram of allocation size')
# plt.stairs(count, bins_count)
# trans = ax.get_xaxis_transform()
# plt.text(51478528, 5000, 'CallStack 132', rotation='vertical', color='red')
# plt.axvline(51478528, color='red')
# plt.text(4198400, 5000, 'CallStack 132', rotation='vertical', color='red')
# plt.axvline(4198400, color='red')
# plt.text(1048576, 5000, 'CallStack 132', rotation='vertical', color='red')
# plt.axvline(1048576, color='red')
# plt.text(2097152, 5000, 'CallStack 132', rotation='vertical', color='red')
# plt.axvline(2097152, color='red')
# plt.text(8388608, 5000, 'CallStack 132', rotation='vertical', color='red')
# plt.axvline(8388608, color='red')
# plt.show()

# print('===========Callstack similarity pairs')
# plt.figure()
# G = nx.Graph()
# G.add_edges_from(np.argwhere(similarityMatrix > 0.75))
# nx.draw_networkx(G)
# plt.show()
# print('Connected pair')
# for i in nx.connected_components(G):
#     print(i)

print('===========Callstack allocation only (remove redundant). Please copy the following output to FindPattern.py')
elemRange = slice(13479, 16899 + 1)
aList = callStackSequence[elemRange]
adSequenceIdPairList = allocationDeallocationSequnceIdPairs[elemRange]
fList = freeSequence[elemRange]

removeRedundantIndex, _ = removeRedundantCallStacks(aList)
_, postRemovalIdMap = removeRedundantCallStacks(
    callStackSequence)  # This array includes all elements, and it useful to map the orignal sequence Id (array index) to sequence Id after removal (array element)


def convertToRemoveRedundantSequenceIds(adPair):
    allocaitonSequenceId, deallocaitonSequenceId = adPair
    return [postRemovalIdMap[allocaitonSequenceId], postRemovalIdMap[deallocaitonSequenceId]]


# These two lines are used in "FindPattern.py"
print('callStackSequence=', [aList[i] for i in removeRedundantIndex])
print('adSequenceIdPairList_keepRedundant=', [adSequenceIdPairList[i] for i in
                                              removeRedundantIndex])  # Note that the sequnce Id here does not consider remove elements
print('adSequenceIdPairList_removeRedundant=', [convertToRemoveRedundantSequenceIds(adSequenceIdPairList[i]) for i in
                                                removeRedundantIndex])  # Note that the sequnce Id here does not consider remove elements
print('postRemovalIdMap=', postRemovalIdMap)

# print(onlyKeepRedundantCallStacks(aList))


exit(-1)
print('===========Callstack allocation with deallocation (remove redundant)')
fList = freeSequence[elemRange]
aListWithDeallocaiton = []
for curLineId in range(len(aList)):
    if callstackInfoList[aList[curLineId]].notRedundant:
        aListWithDeallocaiton.append(aList[curLineId])
    for j in fList[curLineId]:
        if callstackInfoList[abs(j)].notRedundant:
            aListWithDeallocaiton.append(j)
print(aListWithDeallocaiton)
