import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import itertools
import re
from sys import argv

# used below; if 'compare', will run rsquared comparison csv function.
runCompare = argv

loanData = pd.read_csv('LoanStats3d.csv', header=1)

# takes a list of col names and creates a new DF based on included col names.
def partialCopy(data, colArray):
    newData = data.copy(deep=True)

    for col in list(newData.columns.values):
        if col not in colArray:
            del newData[col]

    return newData

# removes single strings and sets to int or float.  good for % or "months", etc.
def strCleaner(data, remItem, objType):
    def remFunc(x):
        x = str(x)
        newStr = x.translate(None, remItem)

        if objType == "float":
            return round(float(newStr) / 100, 4)
        elif objType == "int":
            return int(newStr)
        else:
            return newStr

    newArr = map(remFunc,data)
    return newArr

# data specific - changes employment length strings into int.
def empLengthCleaner(dataCol):
    non_decimal = re.compile(r'[^\d.]+')

    def cleanerMapFunc(item):
        # item = str(item)
        if item == "n/a":
            return 0
        else:
            return int(non_decimal.sub('', item))

    newList = map(cleanerMapFunc, dataCol)

    return newList

# compares all cols in the data against each other *once*.
def compareAll(dfData):
    outputList = []
    headers = ['a', 'b', 'rsquared']

    for a, b in itertools.combinations(dfData.columns.values, 2):
        result = smf.ols(formula='%s ~ %s' % (a, b), data=dfData).fit()
        outputList.append([a, b, result.rsquared])

    df = pd.DataFrame(outputList, columns=headers)

    return df

# builds simple dict - key is distinct items in the col, val is i++
def intDictBuilder(colData, sorted):
    newList = list(set(colData))
    newDict = {}
    counterVar = 0

    if sorted:
        newList = sorted(newList)

    for i in newList:
        newDict[i] = counterVar
        counterVar += 1

    return newDict

# converts categorical data to ascending int; uses intDictbuilder.
def catToInt(data, colName, newColName, sorted):
    
    intDict = intDictBuilder(data[colName], sorted)

    data[newColName] = [intDict[x] for x in data[colName]]

    return data

loanData['intRateFloat'] = strCleaner(loanData['int_rate'], '%', 'float')

loanData = loanData[np.isfinite(loanData['annual_inc'])]
loanData = loanData[np.isfinite(loanData['intRateFloat'])]
loanData = loanData.dropna(subset=['emp_length'])

loanData['empYears'] = empLengthCleaner(loanData['emp_length'])


if runCompare = 'compare':
    numData = loanData.select_dtypes(include=['int64', 'float64'])
    comparedDF = compareAll(numData)
    comparedDF.to_csv('comparisonTable.csv', index=False)

# wantedCols = [
#     'loan_amnt',
#     'term',
#     'intRateFloat',
#     'grade',
#     'emp_length',
#     'home_ownership',
#     'annual_inc',
#     'verification_status',
#     'purpose',
#     'title',
#     'installment',
#     'addr_state',
#     'delinq_2yrs'
# ]

# dfCheck = partialCopy(loanData, wantedCols)

# print dfCheck

# a = pd.scatter_matrix(dfCheck, alpha=0.05, figsize=(14,14), diagonal='hist')

# plt.savefig('scatter_matrix-limited.png')
# plt.show()

# b = pd.scatter_matrix(loanData, alpha=0.05, figsize=(20,20), diagonal='hist')
# plt.savefig('scatter-matrix-all.png')

incCap = 2500000

# loanData = loanData[loanData.annual_inc <= incCap]

loanData.hist(bins=1000, column='annual_inc')
plt.axis([0, 500000, 0, 27500])
plt.savefig('annual_inc-hist.png')

loanData.hist(column='intRateFloat')
plt.savefig('intRate-hist.png')

plt.clf()

plt.scatter(loanData['delinq_2yrs'], loanData['intRateFloat'], alpha=0.05)
plt.savefig('scatter-delinq-int.png')
plt.show()

plt.scatter(loanData['empYears'], loanData['intRateFloat'], alpha=0.1)
plt.savefig('scatter-years-int.png')
plt.show()

plt.scatter(loanData['annual_inc'], loanData['intRateFloat'], alpha=0.05)
plt.axis([0, incCap, 0, 0.35])

income_linspace = np.linspace(0, incCap, 200)
x = pd.DataFrame({'annual_inc': income_linspace})

result0 = smf.ols(formula='intRateFloat ~ annual_inc', data=loanData).fit()
print result0.summary()
print result0.params

plt.plot(income_linspace, result0.params[0] + result0.params[1] * income_linspace, 'r')
plt.show()

result1 = smf.ols(formula='intRateFloat ~ annual_inc + C(home_ownership)', data=loanData).fit()

print result1.summary()
print result1.params

res1Int = result1.params[0]
res1Own = result1.params[1]
res1Rent = result1.params[2]
res1Inc = result1.params[3]

plt.scatter(loanData['annual_inc'], loanData['intRateFloat'], alpha=0.05)
plt.axis([0, incCap, 0, 0.35])

plt.plot(income_linspace, result1.params[0] + result1.params[3] * income_linspace, 'r') # Mortgage
plt.plot(income_linspace, res1Int + res1Inc * income_linspace + res1Own, 'g') # Own
plt.plot(income_linspace, res1Int + res1Inc * income_linspace + res1Rent, 'b') # Rent

plt.savefig('scatter-line.png')
plt.show()

result_2 = smf.ols(formula='intRateFloat ~ annual_inc * C(home_ownership)', data=loanData).fit()

print result_2.summary()
print result_2.params

res2Int = result_2.params[0]
res2Own = result_2.params[1]
res2Rent = result_2.params[2]
res2Inc = result_2.params[3]
res2OwnInc = result_2.params[4]
res2RentInc = result_2.params[5]

plt.scatter(loanData['annual_inc'], loanData['intRateFloat'], alpha=0.05)
plt.axis([0, incCap, 0, 0.35])

plt.plot(income_linspace, res2Int + res2Inc * income_linspace, 'r') # Mortgage
plt.plot(income_linspace, res2Int + res2Inc * income_linspace + res2Own + res2OwnInc * income_linspace, 'g') # Own
plt.plot(income_linspace, res2Int + res2Inc * income_linspace + res2Rent + res2RentInc * income_linspace, 'b') # Rent

plt.savefig('scatter-line-multi.png')
plt.show()

