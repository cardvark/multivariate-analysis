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

print loanData['int_rate']
print loanData['intRateFloat']
print loanData['annual_inc']

# plt.figure()
# loanData.hist(column='annual_inc')
# plt.show()

plt.scatter(loanData['annual_inc'], loanData['intRateFloat'])
plt.axis([0, 2500000, 0, 0.35])

result = smf.ols(formula='intRateFloat ~ annual_inc', data=loanData).fit()

print result.params
print result.params['Intercept']

income_linspace = np.linspace(0, 2500000, 200)

plt.plot(income_linspace, result.params[0] + result.params[1] * income_linspace, 'r')

plt.show()
