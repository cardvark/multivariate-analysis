import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

loanData = pd.read_csv('LoanStats3d.csv', header=1)

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

loanData['intRateFloat'] = strCleaner(loanData['int_rate'], '%', 'float')

loanData = loanData[np.isfinite(loanData['annual_inc'])]
loanData = loanData[np.isfinite(loanData['intRateFloat'])]

# loanData = loanData[loanData.annual_inc <= 3000000]

# loanData['logIncome'] = np.log1p(loanData.annual_inc)

# print loanData['int_rate']
# print loanData['intRateFloat']
# print loanData['annual_inc']

loanData.hist(bins=1000, column='annual_inc')
plt.axis([0, 500000, 0, 27500])
plt.savefig('annual_inc-hist.png')

loanData.hist(column='intRateFloat')
plt.savefig('intRate-hist.png')

plt.scatter(loanData['annual_inc'], loanData['intRateFloat'], alpha=0.05)
plt.axis([0, 2500000, 0, 0.35])

income_linspace = np.linspace(0, 2500000, 200)
x = pd.DataFrame({'annual_inc': income_linspace})

result0 = smf.ols(formula='intRateFloat ~ annual_inc', data=loanData).fit()
print result0.summary()
print result0.params

plt.plot(income_linspace, result0.params[0] + result0.params[1] * income_linspace, 'r')
plt.show()

result1 = smf.ols(formula='intRateFloat ~ annual_inc + C(home_ownership)', data=loanData).fit()

print result1.summary()
print result1.params

# result_3 = smf.ols(formula='intRateFloat ~ 1 + annual_inc + I(annual_inc ** 2.0)', data=loanData).fit()
# print result_3.summary()
# print result_3.params

res1Int = result1.params[0]
res1Own = result1.params[1]
res1Rent = result1.params[2]
res1Inc = result1.params[3]

plt.scatter(loanData['annual_inc'], loanData['intRateFloat'], alpha=0.05)
plt.axis([0, 2500000, 0, 0.35])

plt.plot(income_linspace, result1.params[0] + result1.params[3] * income_linspace, 'r') # Mortgage
plt.plot(income_linspace, res1Int + res1Inc * income_linspace + res1Own, 'g') # Own
plt.plot(income_linspace, res1Int + res1Inc * income_linspace + res1Rent, 'b') # Rent
# plt.plot(income_linspace, result.predict(x), 'r', alpha=0.9)
# plt.plot(x.annual_inc, result_2.predict(x), 'g')

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
plt.axis([0, 2500000, 0, 0.35])

plt.plot(income_linspace, res2Int + res2Inc * income_linspace, 'r') # Mortgage
plt.plot(income_linspace, res2Int + res2Inc * income_linspace + res2Own + res2OwnInc * income_linspace, 'g') # Own
plt.plot(income_linspace, res2Int + res2Inc * income_linspace + res2Rent + res2RentInc * income_linspace, 'b') # Rent

plt.savefig('scatter-line-multi.png')
plt.show()