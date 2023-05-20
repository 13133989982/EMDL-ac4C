import numpy as np

def one_hot1(data, windows=50):
    # define input string
    data = data
    length = len(data)
    print("length:",length)
    # define empty array

    # data_X = np.zeros((length, 2*windows+1, 4))
    data_X = np.zeros((length, windows, 5))
    data_Y = []
    for i in range(length):
        # print("切片前：", data[i])
        x = data[i].split(",") #通过制定分隔符对字符串进行切片
        # x = data[i].split(",")  # 通过制定分隔符对字符串进行切片
        # print("分隔符切片后：x=", x)
        # get label
#         data_Y.append(int(x[2]))  #ValueError: invalid literal for int() with base 10:
#         print("x_label:", x[2])
#         print("x[2]:", x[2])

        data_Y.append(int(float(x[1])))
        # define universe of possible input values
        # nucl = 'ACGTNBDEFHIJKLMNOPQRSU'
        nucl = 'ACGTNDEFHIKLMPQRSVWY-BJOUXZ'
        # define a mapping of chars to integers
        char_to_int = dict((c, i) for i, c in enumerate(nucl))
        # integer encode input data
        # xx = x[1]
        # print("xx:", xx)
        # xxx = xx[1:-1]
        # print("xxx:", xxx)
        # integer_encoded = [char_to_int[char] for char in xxx]
        # print("x[0]:", x[0])
        # print("x[0] type:", type(x[0]))

        integer_encoded = [char_to_int[char] for char in x[0]]
        # print("integer_encoded:",integer_encoded)
        # one hot encode
        j = 0
        for value in integer_encoded:
#             if value in [21, 22, 23, 24, 25, 26]:
#             if value in [4, 5, 6, 7, 8, 9]:
            if value in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,19, 20, 21, 22, 23, 24, 25, 26]:
                # for k in range(5):
                for k in range(5):
                    data_X[i][j][k] = 0.05
            else:
                data_X[i][j][value] = 1.0
#             data_X[i][j][value] = 1.0
#             print("value:", value)
#             print("data_X:", data_X[i][j][value])
            j = j + 1
    data_Y = np.array(data_Y)
    # print("onehot data_Y output:", data_X)

    return data_X, data_Y


def CKSNAP(fastas, gap=3, **kw):
    if gap < 0:
        print('Error: the gap should be equal or greater than zero' + '\n\n')
        return 0
#     if check_sequences.get_min_sequence_length(fastas) < gap + 2:
#         print('Error: all the sequence length should be larger than the (gap value) + 2 = ' + str(gap + 2) + '\n\n')
#         return 0
    AA = kw['order'] if kw['order'] != None else 'ACGT'
    encodings = []
    aaPairs = []
    for aa1 in AA:
        for aa2 in AA:
            aaPairs.append(aa1 + aa2)
#     print("aaPairs:", aaPairs)
#  aaPairs:['AA', 'AC', 'AG', 'AT', 'CA', 'CC', 'CG', 'CT', 'GA', 'GC', 'GG', 'GT', 'TA', 'TC', 'TG', 'TT']

#     header = ['#', 'label']
#     for g in range(gap + 1):
#         for aa in aaPairs:
#             header.append(aa + '.gap' + str(g))
#     encodings.append(header)
    for i in fastas:  #fastas = [name, sequence, label, label_train]
#         name, sequence, label = i[0], i[1], i[2]
#         print("i:", i)
        sequence1 = i[0]
        label1 = i[1]
        code = []
#         code = [name, label]
        for g in range(gap + 1):
            myDict = {}
            for pair in aaPairs:
                myDict[pair] = 0
            sum = 0
            for index1 in range(len(sequence1)):
                index2 = index1 + g + 1
                if index1 < len(sequence1) and index2 < len(sequence1) and sequence1[index1] in AA and sequence1[
                    index2] in AA:
                    myDict[sequence1[index1] + sequence1[index2]] = myDict[sequence1[index1] + sequence1[index2]] + 1
                    sum = sum + 1
            for pair in aaPairs:
                code.append(myDict[pair] / sum)
                # print(len(code))
        encodings.append(code)
    return encodings


def CKSNAP1(data):
    import numpy as np
    train_data = data
    train_data1 = []
    data_y = []
    for i in range(len(train_data)):
        x = train_data[i].split(',')  # 通过制定分隔符对字符串进行切片
        sequence = re.sub('-', '', x[0])
        label = x[1]
        data_y.append(label)
        train_data1.append([sequence, label])

    kw = {'order': 'ACGT'}
    encodings = CKSNAP(train_data1, gap=2, **kw)
    encodings = np.array(encodings)
    data_y = np.array(data_y)
    return encodings, data_y


#########################################################3
def kmerArray(sequence, k):
    kmer = []
    for i in range(len(sequence) - k + 1):
        kmer.append(sequence[i:i + k])
    return kmer

def Kmer(fastas, k=2, type="DNA", upto=False, normalize=True, **kw):
    import re
    import itertools
    from collections import Counter
    encoding = []
    data_Y = []
    header = ['#', 'label']
    NA = 'ACGT'
    if type in ("DNA", 'RNA'):
        NA = 'ACGT'
    else:
        NA = 'ACDEFGHIKLMNPQRSTVWY'

    if k < 1:
        print('Error: the k-mer value should larger than 0.')
        return 0

    if upto == True:
        ke = []
        for tmpK in range(1, k + 1):
            for kmer in itertools.product(NA, repeat=tmpK):
                ke.append(''.join(kmer))
                header.append(''.join(kmer))
        # encoding.append(header)
        print("kmer:", ke)
        for i in fastas:
            i = i.split(",")  # 通过制定分隔符对字符串进行切片
            data_y, sequence, = i[1], re.sub('-', '', i[0])
            data_Y.append(data_y)
            count = Counter()
            for tmpK in range(1, k + 1):
                kmers = kmerArray(sequence, tmpK)
                count.update(kmers)
                if normalize == True:
                    for key in count:
                        if len(key) == tmpK:
                            count[key] = count[key] / len(kmers)
            # code = [name,]
            code = []
            for j in range(2, len(header)):
                if header[j] in count:
                    code.append(count[header[j]])
                else:
                    code.append(0)
            encoding.append(code)
    encoding = np.array(encoding)
    data_Y = np.array(data_Y)

    return encoding, data_Y

import re
def TriNcleotideComposition(sequence, base):
    trincleotides = [nn1 + nn2 + nn3 for nn1 in base for nn2 in base for nn3 in base]
    tnc_dict = {}
    for triN in trincleotides:
        tnc_dict[triN] = 0
    for i in range(len(sequence) - 2):
        if re.search('-', sequence[i:i + 3]):
            pass
        else:
            # print("@@@ :", sequence[i:i + 3])
            tnc_dict[sequence[i:i + 3]] += 1
    for key in tnc_dict:
        tnc_dict[key] /= (len(sequence) - 2)
    return tnc_dict

def PseEIIP1(fastas):
    #     for i in fastas:
    #         if re.search('[^ACGT-]', i[1]):
    #             print('Error: illegal character included in the fasta sequences, only the "ACGT-" are allowed by this PseEIIP scheme.')
    #             return 0
    base = 'ACGT'
    EIIP_dict = {
        'A': 0.1260,
        'C': 0.1340,
        'G': 0.0806,
        'T': 0.1335,
    }

    # 三核苷酸排列组合
    trincleotides = [nn1 + nn2 + nn3 for nn1 in base for nn2 in base for nn3 in base]

    #     print("trincleotides of length:", len(trincleotides))
    #     print("trincleotides:", trincleotides)

    EIIPxyz = {}

    # 计算EIIPxyz
    for triN in trincleotides:
        EIIPxyz[triN] = EIIP_dict[triN[0]] + EIIP_dict[triN[1]] + EIIP_dict[triN[2]]

    encodings = []
    data_y = []
    # header = ['#'] + trincleotides

    #     print("header:", header)

    #     encodings.append(header)
    # data_Y = []

    for i in range(len(fastas)):
        #         name, sequence = i[0], re.sub('-', '', i[1])
        x = fastas[i].split(",")  # 通过制定分隔符对字符串进行切片
        #         print("x:",x)
        # sequence = x[0]
        sequence = re.sub('-', '', x[0])

        data_y.append(int(float(x[1])))
        #         code = [name]
        trincleotide_frequency = TriNcleotideComposition(sequence, base)  # 每个三核苷酸的频率
        #         code = code + [EIIPxyz[triN] * trincleotide_frequency[triN] for triN in trincleotides]
        code = [EIIPxyz[triN] * trincleotide_frequency[triN] for triN in trincleotides]
        encodings.append(code)
    #     print("############################encodings:", encodings)
    #     print("############################encodings[0]:", len(encodings[0]))
    # data_Y = np.array(data_Y)

    encodings = np.array(encodings)
    # encodings = np.reshape(encodings, (len(encodings), 64, 1))
    data_y = np.array(data_y)
    # return encodings, data_Y
    # print("PSE##############################3:", encodings)

    return encodings, data_y
