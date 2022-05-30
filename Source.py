from sklearn.cluster import KMeans
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("Mall_Customers.csv")  # đọc file csv
print(df.head())  # in ra bản ghi đầu tiên của dataframe

x = df.iloc[:, [3, 4]].values
wcss = []  # wcss: within cluster sum of square

for k in range(1, 11):
    km = KMeans(n_clusters=k)
    km.fit(x)  # được sử dụng để tạo các tham số mô hình
    # tổng bình phương khoảng cách của các mẫu đến trung tâm cụm gần nhất
    wcss.append(km.inertia_)  # hoặc tổng sai số bình phương

plt.plot(range(1, 11), wcss)
plt.title("The elbow point graph")
plt.xlabel("Values of k")
plt.ylabel("wcss")
plt.show()


km = KMeans(n_clusters=5)

y_predicted = km.fit_predict(x)
# tính toán các trung tâm cụm và đự đoán chỉ số cụm


# vẽ biểu đồ phân tán
plt.scatter(x[y_predicted == 0, 0], x[y_predicted == 0, 1], color='green')
plt.scatter(x[y_predicted == 1, 0], x[y_predicted == 1, 1], color='red')
plt.scatter(x[y_predicted == 2, 0], x[y_predicted == 2, 1], color='blue')
plt.scatter(x[y_predicted == 3, 0], x[y_predicted == 3, 1], color='cyan')
plt.scatter(x[y_predicted == 4, 0], x[y_predicted == 4, 1], color='black')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color='m')

plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(['cum 1', 'cum 2', 'cum 3', 'cum 4', 'cum 5', 'center'])

plt.show()
