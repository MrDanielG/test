import kmeans_funciones as kmf

opt = input('Desea correr el programa 1(Pajaro) o 2 (Graficas): ')
opt = int(opt)

if opt == 1:
    #Carguemos la imagen
    data_imagen = kmf.loadmat('data/bird_small_kmeans.mat')
    A = data_imagen['A']
    A.shape

    # Normalicemos los rangos de los valores
    A = A/255.

    # Reshape el arreglo
    X = kmf.np.reshape(A, (A.shape[0] * A.shape[1], A.shape[2]))

    #Iniciar los centroides aleatoriamente
    initial_centroids = kmf.init_centroids(X, 16)

    #correr el algoritmo
    idx, centroids = kmf.run_k_means(X, initial_centroids, 10)

    #Obtener los centroides mas vercanos una vez mas
    idx = kmf.find_closest_centroids(X, centroids)

    #Mapear cada pixel al calor del centroide
    X_recovered = centroids[idx.astype(int),:]

    #Recambiar a las dimensiones originales
    X_recovered = kmf.np.reshape(X_recovered, (A.shape[0], A.shape[1], A.shape[2]))

    kmf.plt.imshow(X_recovered)
    kmf.plt.savefig('pajaro_chico.png')

else:
    #Carguemos una base de datos del clima
    data = kmf.loadmat('data/clustering_colors.mat')
    X = data['X']

    #Propongamos unos centroides iniciales (en principio esto debe ser aleatorio)
    initial_centroids = initial_centroids = kmf.np.array([[3, 3], [6, 2], [8, 5]])

    #Estimemos donde estan los centroides mas cercanos
    idx = kmf.find_closest_centroids(X, initial_centroids)
    print(idx[0:500])

    kmf.compute_centroids(X, idx, 3)

    #corremos el algoritmo (nuestro resultado es las posiciones de los centroides)
    idx, centroids = kmf.run_k_means(X, initial_centroids, 10)
    #print(idx)
    print(centroids)

    #hacer graficas
    #Aqui definimos nuestros cluster
    cluster1 = X[kmf.np.where(idx == 0)[0],:]
    cluster2 = X[kmf.np.where(idx == 1)[0],:]
    cluster3 = X[kmf.np.where(idx == 2)[0],:]

    fig, ax = kmf.plt.subplots(figsize=(12,8))
    ax.scatter(cluster1[:,0], cluster1[:,1], s=30, color='r', label='Cluster 1')
    ax.scatter(cluster2[:,0], cluster2[:,1], s=30, color='g', label='Cluster 2')
    ax.scatter(cluster3[:,0], cluster3[:,1], s=30, color='b', label='Cluster 3')
    ax.legend()

    kmf.plt.xlabel('Diferencia de temperatura')
    kmf.plt.ylabel('Diferencia de presion')
    kmf.plt.savefig("kmeans.pdf")
