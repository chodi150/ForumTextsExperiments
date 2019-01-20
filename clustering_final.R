library(stats)
library(slam)
library(e1071)
library(xgboost)
library(ROCR)
library(dismo)
library(cluster)
library(e1071)
library(caret)
library(mlbench)
library(pROC)
library(proxy)
library(fossil)
library(factoextra)
library(rlist)

path <- getwd()
file1 <- "/forum_haszysz/glove_window_size_5_vec_dim_100_pruned_haszysz_WAZONE_TF11-01-2019-16-19.csv"
file2 <- "/forum_haszysz/dict_size_807_tfidf_haszyszdobry12-01-2019-17-49.csv"
file3 <- "/forum_bmw/glove_window_size_5_vec_dim_100_PRUNED_BMW_0_5_PROC_POS11-01-2019-17-07.csv"
file4 <- "/forum_bmw/dict_size_913_tfidf_pruned_BMW_tfidf12-01-2019-03-28.csv"
charts_path <- paste0(path,"/charts/")
silhouette_filename <- "silhouette.png"
points_clusters_filename <- "points.png"
desired_attempt_size <- 4000
krange <- 2:15


prepareDataFrame <- function(path) {
    data_frame <- read.csv(file = path, header = TRUE, sep = ";")
    data_frame$X <- NULL
   # classes <- data_frame$belongs_to
    data_frame$belongs_to <- NULL
    return (data_frame)
}

prepareDataFrameAndClasses <- function(path) {
    data_frame <- read.csv(file = path, header = TRUE, sep = ";")
    data_frame$X <- NULL
    classes <- data_frame$belongs_to
    if (is.null(classes)) {
        classes <- data_frame$category
        data_frame$category <- NULL
    }
    data_frame$belongs_to <- NULL
    res <- list(data_frame, classes)
    names(res) <- c("df", "classes")
    return(res)
}

performClusteringPAM <- function(data_frame, range_k, attempt_size, folder_name) {
    silhouettes <- list()
    index <- 1:nrow(data_frame)
    division_param <- nrow(data_frame) / desired_attempt_size

    attempt_index <- sample(index, trunc(length(index) / division_param))
    data_frame_attempt <- data_frame[attempt_index,]
   # classes_attempt <- classes[attempt_index]

    dist.matrix = proxy::dist(data_frame_attempt, method = "cosine")

    for (k in range_k) {
        print(paste("Clustering for k:", k))
        pam_clustering <- pam(dist.matrix, k, trace.lev = 0)
        silh <- silhouette(pam_clustering$clustering, dist = dist.matrix)
        silh_info <- summary(silh)
        silhouettes <- list.append(silhouettes, silh_info)
        png(paste0(charts_path, folder_name, k, silhouette_filename), width = 512, height = 512)
        plot(silh, col = 1:k, border = NA, main = paste0("Silhouette plot for k=", k))
        dev.off()
        points <- cmdscale(dist.matrix, k = k) # Running the PCA
        png(paste0(charts_path, folder_name, k, points_clusters_filename), width = 512, height = 512)
        plot(points, main = 'PAM Clustering', col = as.factor(pam_clustering$clustering), mai = c(0, 0, 0, 0), mar = c(0, 0, 0, 0), xaxt = 'n', yaxt = 'n', xlab = '', ylab = '')
        dev.off()
    }
    return (silhouettes)

}

write_results_to_csv <- function(list_silhs, folder) {
    fname <- paste0(folder, "results.csv")
    write("avg; widths", file = fname)
    for (i in 1:length(list_silhs)) {
        width <- list_silhs[[i]]$avg.width
        both <- list_silhs[[i]]$clus.avg.widths
        empty <- ""
        for (j in 1:length(both)) {
            empty <- paste(empty, ";", both[[j]])
        }
        write(paste0(width, ";", empty), fname, append = TRUE)
    }
}


plotAvgSilh <- function(results1, results2, results3, color) {

    avgs <- list()
    sds <- list()
    ks <- list()
    for (i in 1:length(results1)) {
        avgs_comb = c(results1[[i]]$avg.width, results2[[i]]$avg.width, results3[[i]]$avg.width)
        avg = mean(avgs_comb)
        sd = sd(avgs_comb)
        avgs <- list.append(avgs, avg)
        sds <- list.append(sds, sd)
        ks <- list.append(ks, length(results1[[i]]$clus.sizes))
    }

    plot(unlist(ks), unlist(avgs), xlab = "Number of clusters", ylab = "Silhouette width",
         col = color)
    arrows(unlist(ks), col = color,
           unlist(avgs) - unlist(sds), unlist(ks), unlist(avgs) + unlist(sds), code = 3, length = 0.02, angle = 90)
    lines(unlist(ks), unlist(avgs), col = color)
    print(max(unlist(avgs)))
    print(unlist(sds))
}

performClustering <- function(data_frame, classes, niter, desired_attempt_size, k) {

    rands_pam <- list()
    rands_hac <- list()
    sili_infos_pam <- list()
    sili_infos_hac <- list()
    for (i in 1:niter) {
        print(paste("Iteration", i))
        index <- 1:nrow(data_frame)
        division_param <- nrow(data_frame) / desired_attempt_size

        attempt_index <- sample(index, trunc(length(index) / division_param))
        data_frame_attempt <- data_frame[attempt_index,]
        classes_attempt <- classes[attempt_index]
        levels(classes_attempt) <- c(1:2)
        classes_attempt <- as.numeric(classes_attempt)

        dist.matrix = proxy::dist(data_frame_attempt, method = "cosine")


        clust <- hclust(dist.matrix, method = "ward.D", members = NULL)
        clustering <- cutree(clust, k = k)
        silh <- silhouette(clustering, dist = dist.matrix)
        sili_infos_hac[[i]] <- summary(silh)
        rands_hac[[i]] <- rand.index(clustering, classes_attempt)

        pam_clustering <- pam(dist.matrix, k, trace.lev = 1)
        clustering <- pam_clustering$clustering
        silh <- silhouette(clustering, dist = dist.matrix)
        sili_infos_pam[[i]] <- summary(silh)
        rands_pam[[i]] <- rand.index(clustering, classes_attempt)

    }

    result <- list(rands_pam, sili_infos_pam, rands_hac, sili_infos_hac)
    names(result) <- c("rands_pam", "sili_infos_pam", "rands_hac", "sili_infos_hac")
    return(result)
}
printClusterStats <- function(results) {
    print("PAM:")
    print(paste("avg rand", mean(unlist(results$rands_pam))))
    print(paste("sd rand", sd(unlist(results$rands_pam))))
    printSilhouette(results$sili_infos_pam)

    print("HAC:")
    print(paste("avg rand", mean(unlist(results$rands_hac))))
    print(paste("sd rand", sd(unlist(results$rands_hac))))
    printSilhouette(results$sili_infos_hac)
}


printSilhouette <- function(sils) {
    avgs <- list()
    for (i in 1:length(sils)) {
        avgs <- list.append(avgs, sils[[i]]$avg.width)
    }
    print(mean(unlist(avgs)))
    print(sd(unlist(avgs)))
}

## Determining k for each dataset
data_frame <- prepareDataFrame(paste0(path,file1))
hasz_glove1 <- performClusteringPAM(data_frame, krange, desired_attempt_size, "hasz/glove1/")
hasz_glove2 <- performClusteringPAM(data_frame, krange, desired_attempt_size, "hasz/glove2/")
hasz_glove3 <- performClusteringPAM(data_frame, krange, desired_attempt_size, "hasz/glove3/")
write_results_to_csv(hasz_glove1, paste0(charts_path, "/hasz/glove1/") )
write_results_to_csv(hasz_glove2, paste0(charts_path, "/hasz/glove2/"))
write_results_to_csv(hasz_glove3, paste0(charts_path, "/hasz/glove3/"))

data_frame <- prepareDataFrame(paste0(path, file3))
bmw_glove1 <- performClusteringPAM(data_frame, krange, desired_attempt_size, "bmw/glove1/")
bmw_glove2 <- performClusteringPAM(data_frame, krange, desired_attempt_size, "bmw/glove2/")
bmw_glove3 <- performClusteringPAM(data_frame, krange, desired_attempt_size, "bmw/glove3/")
write_results_to_csv(bmw_glove1, paste0(charts_path, "/bmw/glove1/"))
write_results_to_csv(bmw_glove2, paste0(charts_path, "/bmw/glove2/"))
write_results_to_csv(bmw_glove3, paste0(charts_path, "/bmw/glove3/"))

data_frame <- prepareDataFrame(paste0(path, file2))
hasz_tfidf1 <- performClusteringPAM(data_frame, krange, desired_attempt_size, "hasz/tf1/")
hasz_tfidf2 <- performClusteringPAM(data_frame, krange, desired_attempt_size, "hasz/tf2/")
hasz_tfidf3 <- performClusteringPAM(data_frame, krange, desired_attempt_size, "hasz/tf3/")

write_results_to_csv(hasz_tfidf1, paste0(charts_path, "/hasz/tf1/"))
write_results_to_csv(hasz_tfidf2, paste0(charts_path, "/hasz/tf2/"))
write_results_to_csv(hasz_tfidf3, paste0(charts_path, "/hasz/tf3/"))

data_frame <- prepareDataFrame(paste0(path, file4))
bmw_tfidf1 <- performClusteringPAM(data_frame, krange, desired_attempt_size, "bmw/tf1/")
bmw_tfidf2 <- performClusteringPAM(data_frame, krange, desired_attempt_size, "bmw/tf2/")
bmw_tfidf3 <- performClusteringPAM(data_frame, krange, desired_attempt_size, "bmw/tf3/")

write_results_to_csv(bmw_tfidf1, paste0(charts_path, "/bmw/tf1/"))
write_results_to_csv(bmw_tfidf2, paste0(charts_path, "/bmw/tf2/"))
write_results_to_csv(bmw_tfidf3, paste0(charts_path, "/bmw/tf3/"))



plotAvgSilh(hasz_glove1, hasz_glove2, hasz_glove3,1)
plotAvgSilh(bmw_glove1, bmw_glove2, bmw_glove3, 1)
plotAvgSilh(hasz_tfidf1, hasz_tfidf2, hasz_tfidf2, 1)
plotAvgSilh(bmw_tfidf1, bmw_tfidf2, bmw_tfidf2, 1)


# Test execution for determined k's

result <- prepareDataFrameAndClasses(paste0(path, file1))
data_frame <- result$df
classes <- result$classes
clus_data <- performClustering(data_frame, classes, 10, 4000)
clus_glove_hasz <- clus_data
printClusterStats(clus_glove_hasz)



result <- prepareDataFrameAndClasses(paste0(path, file2))
data_frame <- result$df
classes <- result$classes
clus_tfidf_hasz <- performClustering(data_frame, classes, 10, 4000,15)
printClusterStats(clus_tfidf_hasz)

result <- prepareDataFrameAndClasses(paste0(path, file3))
data_frame <- result$df
classes <- result$classes
clus_glove_bmw <- performClustering(data_frame, classes, 10, 4000, 3)
printClusterStats(clus_glove_bmw)

result <- prepareDataFrameAndClasses(paste0(path, file4))
data_frame <- result$df
classes <- result$classes
clus_tfidf_bmw <- performClustering(data_frame, classes, 10, 4000, 15)
printClusterStats(clus_tfidf_bmw)
