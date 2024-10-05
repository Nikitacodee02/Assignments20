#!/usr/bin/env python
# coding: utf-8

# State the Hypotheses:
# 
# Null Hypothesis (H0): There is no significant association between the type of device purchased and the customer satisfaction level. In other words, the satisfaction level is independent of the device type.
# 
# Alternative Hypothesis (H1): There is a significant association between the type of device purchased and the customer satisfaction level.

# In[1]:


import numpy as np
from scipy.stats import chi2_contingency, chi2

# Step 1: Create the Contingency Table
data = np.array([
    [50, 70],  # Very Satisfied
    [80, 100], # Satisfied
    [60, 90],  # Neutral
    [30, 50],  # Unsatisfied
    [20, 50]   # Very Unsatisfied
])

# Step 2: Calculate Chi-Square Statistic
chi2_stat, p_value, dof, expected = chi2_contingency(data)

# Step 3: Determine the Critical Value
alpha = 0.05
critical_value = chi2.ppf(1 - alpha, dof)

# Step 4: Make Decision
decision = "Reject the null hypothesis" if chi2_stat > critical_value else "Fail to reject the null hypothesis"

# Output
print(f"Chi-Square Statistic: {chi2_stat:.4f}")
print(f"P-Value: {p_value:.4f}")
print(f"Degrees of Freedom: {dof}")
print(f"Critical Value: {critical_value:.4f}")
print(f"Decision: {decision}")


# ####The Chi-Square test was used to determine if there is a significant relationship between the type of smart home device purchased and customer satisfaction levels. The test results showed a Chi-Square statistic greater than the critical value at a 0.05 significance level. This means we reject the null hypothesis and conclude that there is a significant association between the type of device and customer satisfaction. In other words, customer satisfaction levels vary depending on whether they bought a Smart Thermostat or a Smart Light.

# # hypothesis testing
# 

# In[ ]:


State the Hypotheses:

Null Hypothesis (H0): The weekly operating costs have not increased; 
    i.e., the observed cost matches the theoretical model. Mathematically:μ=$1,000+$5×600.

Alternative Hypothesis (H1): The weekly operating costs have increased;
    i.e., the observed cost is higher than the theoretical model.


# In[2]:


import scipy.stats as stats

# Given data
sample_mean = 3050
number_of_units = 600
standard_deviation = 5 * 25  
sample_size = 25
alpha = 0.05

# Calculate the theoretical mean cost
theoretical_mean = 1000 + 5 * number_of_units

# Calculate the test statistic
test_statistic = (sample_mean - theoretical_mean) / (standard_deviation / (sample_size ** 0.5))

# Determine the critical value for a one-tailed test at alpha = 0.05
critical_value = stats.norm.ppf(1 - alpha)

# Print results
print(f"Test Statistic: {test_statistic:.2f}")
print(f"Critical Value: {critical_value:.2f}")

# Make a decision
if test_statistic > critical_value:
    print("Reject the null hypothesis: There is strong evidence that the weekly operating costs are higher than the model suggests.")
else:
    print("Fail to reject the null hypothesis: There is not enough evidence to support the claim that the weekly operating costs are higher than the model suggests.")


# INTERPRETATION
# 
# The test statistic was calculated to be -38, and the critical value for a one-tailed test at the 0.05 significance level was 1.64. Since the test statistic exceeds the critical value, we reject the null hypothesis. This means there is strong evidence to support the restaurant owners' claim that their weekly operating costs are indeed higher than what the cost model suggests.
