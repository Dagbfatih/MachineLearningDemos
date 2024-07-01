
# Machine Learning

## Links

#### Data
https://files.btkakademi.gov.tr/259_VERI_BILIMI_VE_MAKINE_OGRENMESI/Veri_Setleri.zip

#### Presentations
https://files.btkakademi.gov.tr/259_VERI_BILIMI_VE_MAKINE_OGRENMESI/Egitmen_Sunumlari.zip

## Terminology
- **Predictive**: estimate, predict
- **Wisdom**: wisdom -> judge -> judgment -> judge -> manage
- **Knowledge**: how
- **Information**: what, when, etc.
- **Data**: data

## Normalization
- **Min-max normalization**: Scale data between 18-70 years old to 0-1 range. This algorithm fixes all data within a range such that 18 -> 0 and 70 -> 1.
- **Z-score normalization**: If the max and min values in noisy data are too extreme, other data may get squeezed. We can overcome this with z-score normalization. Z-score normalization normalizes the data to have a mean of 0 and a standard deviation of 1 and -1.
- **Decimal normalization**: 

## Induction and Deduction
- **Induction**: induction
- **Deduction**: deduction

## Data Cleaning Steps
1. Rename column names
2. Remove useless data/columns
3. Fill in missing data (height, weight, etc.)

## Classical Machine Learning
Feature inference is done and determined by humans. For example, in the case of ayran, soda, and water, I say the features to look at are color and bubbles and provide a certain amount of labeled data to the software.

## Deep Learning
I provide it with an excessive amount of labeled data, and the only difference from CML is that it determines the features itself.

## Perceptron - Artificial Neural Network
[Link](https://towardsdatascience.com/the-mostly-complete-chart-of-neural-networks-explained-3fb6f2367464)

## Regression Analysis
- **Independent variable**: x
- **Dependent variable**: y
- f(x) = y

### Types of Regression Analysis
1. **Simple Regression Analysis**
   - 1 dependent & 1 independent variable

2. **Multiple Regression Analysis**
   - 1 dependent & multiple independent variables

3. **Multiple Variable (Polynomial) Regression Analysis**
   - Multiple dependent variables

### Linear Regression Analysis
- If the relationship between variables is linear

### Non-Linear Regression Analysis
- If the relationship between variables is not linear

### Simple Regression Model
\[ y = β0 + β1x + ϵ \]
- β0: constant value. When x equals 0, represents y value. (Intercept)
- β1: Regression coefficient. Like 'a' in ax + b
- ϵ: Random error term. Assumes the dependent value includes an error. Generally ignored.

## Note
Manhattan distance means the total of the vertical and horizontal path between two points on the coordinate system. The Euclidean path also means the linear path length on the coordinate system.

## Regularization
Used to figure out the overfitting problem. Overfitting generally occurs when the data at hand is noisy.

## Principal Component Analysis (PCA)
In a survey with 50 questions, 15 questions measure income level, 10 questions measure something else, and if we put this data into PCA, we can see that there are only 3-4 main data points. It simplifies and reduces a lot of data to a small number.

Before doing this, the data must be standardized and normalized.

## Data Separation
**t-SNE and PCA are used to separate data**

## Kernel Trick - Support Vector Machine

## Classification

# Example of Confusion Matrix in Machine Learning

Machine learning, especially in statistical classification problems, often uses a confusion matrix, also known as an error matrix, to visualize the performance of an algorithm. This special table is typically used for supervised learning.

## Example

Let's create an example.

- Suppose we have 13 images. Eight of these images are of cats, and five are of dogs.
- Let the cats be classified as 1 and the dogs as 0.
- If we list the true classes of these images as follows: [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
- And we provide these images to a classifier, which then gives the following predictions: [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1]
- To interpret these two lists, we can create a table.

### Confusion Matrix

|               | Actual Cat | Actual Dog |
|---------------|-------------|-------------|
| Predicted Cat |      5      |      2      |
| Predicted Dog |      3      |      3      |

We had 13 images and we got 8 images right. So the accuracy is **8/13**.

**Contingency**: Durumsallık - BTK Akademi Kurs, 9. section, 2. video, 2. minute

