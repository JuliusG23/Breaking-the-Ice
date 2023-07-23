####################################
# Preparation
####################################

# load packages
library(ggplot2)
library(plyr)
library(stargazer)
library(dplyr)
library(tidyverse)
library(reshape)
library(readxl)

# set working directory
setwd("...")


####################################
# Find optimal weights
####################################

# load data
entpop <- read.csv("weights_entpop.csv")
greedypop <- read.csv("weights_greedypop.csv")

# group data by different weight combinations
# entpop
entpop_optimal_weights <- data.frame(entpop %>%
  group_by(Weight.popularity, Weight.entropy0) %>%
  summarise(RMSE = round(mean(RMSE), 5)))

# greedypop
greedypop_optimal_weights <- data.frame(greedypop %>%
  group_by(Weight.popularity, Weight.greedy.extent) %>%
  summarise(RMSE = round(mean(RMSE), 5)))

# extract optimal weights and only keep relevant observations
# entpop
entpop_optimal_weights[entpop_optimal_weights$RMSE == min(entpop_optimal_weights$RMSE), ]

# greedypop
greedypop_optimal_weights[greedypop_optimal_weights$RMSE == min(greedypop_optimal_weights$RMSE), ]

# only keep relevant values
# entpop
entpop_results <- entpop[entpop$Weight.popularity == entpop_optimal_weights[entpop_optimal_weights$RMSE == min(entpop_optimal_weights$RMSE), ]$Weight.popularity, ]
entpop_results <- entpop_results[entpop_results$Weight.entropy0 == entpop_optimal_weights[entpop_optimal_weights$RMSE == min(entpop_optimal_weights$RMSE), ]$Weight.entropy0, ]

# greedypop
greedypop_results <- greedypop[greedypop$Weight.popularity == greedypop_optimal_weights[greedypop_optimal_weights$RMSE == min(greedypop_optimal_weights$RMSE), ]$Weight.popularity, ]
greedypop_results <- greedypop_results[greedypop_results$Weight.greedy.extent == greedypop_optimal_weights[greedypop_optimal_weights$RMSE == min(greedypop_optimal_weights$RMSE), ]$Weight.greedy.extent, ]

# create table
#entpop
table_entpop_optimal_weights <- data.frame(entpop_optimal_weights %>%
             pivot_wider(names_from = Weight.popularity, values_from = RMSE))

# greedypop
table_greedypop_optimal_weights <- data.frame(greedypop_optimal_weights %>%
                                             pivot_wider(names_from = Weight.popularity, values_from = RMSE))

# export to latex
stargazer(table_entpop_optimal_weights, summary = F, digits = NA, rownames = F)
stargazer(table_greedypop_optimal_weights, summary = F, digits = NA, rownames = F)

# # use different weights for different set sizes
# # Subset the dataframe by 'Nr..of.shown.items' and find the rows with the lowest RMSE
# entpop_results <- entpop[order(entpop$RMSE), ]  # Sort the dataframe by 'RMSE'
# entpop_results <- entpop_results[!duplicated(entpop_results$Nr..of.shown.items), ]  # Keep only the first occurrence of each 'Nr..of.shown.items'
# 
# greedypop_results <- greedypop[order(greedypop$RMSE), ]  # Sort the dataframe by 'RMSE'
# greedypop_results <- greedypop_results[!duplicated(greedypop_results$Nr..of.shown.items), ]  # Keep only the first occurrence of each 'Nr..of.shown.items'


####################################
# Load and combine remaining data
####################################
items_10 <- read.csv("final_results2_shown_items_10.csv")
items_25 <- read.csv("final_results2_shown_items_25.csv")
items_50 <- read.csv("final_results2_shown_items_50.csv")
items_100 <- read.csv("final_results2_shown_items_100.csv")

# combine data in one df
data <- rbind(items_10, items_25, items_50, items_100, entpop_results[,1:4], greedypop_results[,1:4])

# adjust column name
colnames(data) <- c("Ranking_strategy", "Nr_shown_items", "Nr_cold_users", "RMSE")

# adjust values ranking strategy column
data$Ranking_strategy <- revalue(data$Ranking_strategy, c("Entropy0 popularity strategy" = "Entropy0-popularity strategy",
                                                          "Greedy extent popularity strategy" = "Greedy extent-popularity strategy", 
                                                          "Random popularity strategy" = "Random-popularity strategy"))


####################################
# Visualize data
####################################

# plot data
ggplot(data, aes(x = Nr_shown_items, y = RMSE, color = Ranking_strategy)) +
  geom_point(size = 1.5) +
  geom_line(linewidth = 1) +
  scale_color_brewer(palette = "Set1") +
  theme_light() +
  theme(legend.position = "bottom") +
  labs(x = "Number of shown items to cold user", color = "Ranking strategy") +
  scale_x_continuous(breaks=c(10, 25, 50, 100))

setwd("...")

# Save plot as pdf
dev.print(pdf,           # file type
          "Results.pdf") # Name of the file

setwd("...")

#table to latex


# Adjusting data for table
adjusted_data <- data.frame(data[,c(1,2,4)] %>%
                              pivot_wider(names_from = Nr_shown_items, values_from = RMSE))

adjusted_data$Mean <- rowMeans(adjusted_data[2:5])

# Print the adjusted data frame
stargazer(adjusted_data, summary = F, rownames = F, digits = 5)


####################################
# Compare random strategy to Geurts et al
####################################

# load data
list_mine <- c(t(adjusted_data[1, 2:5]))
list_geurts <- c(0.5547, 0.3992, 0.4951, 0.4539)

# non-parametric t-test
wilcox.test(list_mine, list_geurts)

# bootstrap
result_geurts <- data.frame(matrix(ncol = 4, nrow = 0))
for(i in 1:2000) {
  result_geurts[i,] <- sample(list_geurts, 4, replace = T)
}

result_geurts$mean <- rowMeans(result_geurts)

# plot bootstrap
ggplot(data = result_geurts, aes(mean)) +
  geom_histogram(bins = 10, fill = "blue", color = "white") +
  geom_vline(aes(xintercept = quantile(mean, 0.025),
             color = "95% confidence intervals"), linewidth = 1.5) +
  geom_vline(aes(xintercept = quantile(mean, 1 - 0.025),
                 color = "95% confidence intervals"), linewidth = 1.5) +
  geom_vline(aes(xintercept = mean(list_mine), color =  "RMSE this thesis"), linewidth = 1.5) +
  theme_light() +
  theme(legend.position = "bottom", legend.title=element_blank()) +
  scale_color_manual(values = c( "RMSE this thesis" = "orange", "95% confidence intervals" = "red")) + 
  labs(x= 'Mean', y = "Count")

setwd("...")

# Save plot as pdf
dev.print(pdf,           # file type
          "Bootstrap.pdf") # Name of the file

setwd("...")


####################################
# Compare all strategies to Geurts et al
####################################

# load data
setwd("...")
results_geurts <- read_excel("Results.xlsx")

# adjust data
results_geurts$`Ranking strategies` <- revalue(results_geurts$`Ranking strategies`, c("Random strategy" = "Random strategy Geurts"))
colnames(results_geurts) <- c("Ranking_strategy", "X10", "X25", "X50", "X100", "Mean")
adjusted_data[2:6] <- round(adjusted_data[2:6], 4)

# combine data
combined <- rbind(adjusted_data, results_geurts)

# remove mean
combined <- combined[,1:5]

# create seperate rows for each set size
combined <- combined %>%
  gather(key = "Set_size", value = "RMSE", -Ranking_strategy)

# ANOVA
anova <- aov(RMSE ~ Ranking_strategy, data = combined)
summary(anova)

# pairwise t test
pairwise_test <- pairwise.t.test(as.numeric(combined$RMSE), combined$Ranking_strategy,
                     p.adjust.method= "bonferroni")
pairwise_test

as.matrix(pairwise_test$p.value)
stargazer(as.matrix(pairwise_test$p.value))

# Define color breaks and colors
color_breaks <- c(0, 0.05, 1)  # Adjust the breaks as needed
colors <- c("black", "white")  # Assign black for values below 0.05 and white for others

par(mar=c(6,6,2,2))
par(oma=c(6,6,2,2))
# Create the heatmap plot
heatmap(as.matrix(pairwise_test$p.value),
        col = colors,
        breaks = color_breaks,
        Rowv = NA, # no dendogram
        Colv = NA, # no dendogram
        cexRow=0.8,
        cexCol=0.8,
        scale = "none")

data_melt <- melt(as.matrix(pairwise_test$p.value))

ggplot(data_melt, aes(X1, X2)) +
  geom_tile(aes(fill = value)) +
  scale_fill_gradient(low = "black", high = "white") +
  labs(fill='Bonferroni corrected p-value') +
  theme_light() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),
        axis.title.x=element_blank(), axis.title.y=element_blank()) +
  theme(legend.position = "bottom")

setwd("...")

# Save plot as pdf
dev.print(pdf,           # file type
          "Comparision.pdf") # Name of the file

setwd("...")


####################################
# Sensitivity analysis
####################################

# load data
sensitivity <- read.csv("sensitivity entropy0.csv")
sensitivity <- sensitivity[!duplicated(sensitivity), ]

# adjust column name
colnames(sensitivity) <- c("Ranking_strategy", "Nr_shown_items", "Nr_cold_users", "RMSE", "Sample")

# adjust values ranking strategy column
sensitivity$Ranking_strategy <- revalue(sensitivity$Ranking_strategy,
            c("Entropy0 popularity strategy" = "Entropy0-popularity strategy"))

# plot data entropy0
ggplot(sensitivity[sensitivity$Ranking_strategy == "Entropy0 strategy", ], aes(x = Nr_shown_items, y = RMSE)) +
  geom_point(aes(group = as.factor(Sample)), size = 1.5, color = "grey") +
  geom_line(aes(group = as.factor(Sample), color = "Additional samples"), linewidth = 1) +
  
  geom_point(data = data[data$Ranking_strategy == "Entropy0 strategy", ], aes(x = Nr_shown_items, y = RMSE), size = 1.5, color = "orange") +
  geom_line(data = data[data$Ranking_strategy == "Entropy0 strategy", ], aes(x = Nr_shown_items, y = RMSE, linetype = "Original"), linewidth = 1, color = "orange") +
  
  theme_light() +
  theme(legend.position = "bottom", legend.title=element_blank()) + 
  labs(x = "Number of shown items to cold user", title = "Entropy0 strategy") +
  scale_x_continuous(breaks = c(10, 25, 50, 100)) +
  scale_linetype_manual(values = "solid", labels = c("Original")) +
  scale_color_manual(values = "grey")


setwd("...")

# Save plot as pdf
dev.print(pdf,           # file type
          "Sensitivity_entropy0.pdf") # Name of the file

# plot data entropy0-popularity
ggplot(sensitivity[sensitivity$Ranking_strategy == "Entropy0-popularity strategy", ], aes(x = Nr_shown_items, y = RMSE)) +
  geom_point(aes(group = as.factor(Sample)), size = 1.5, color = "grey") +
  geom_line(aes(group = as.factor(Sample), color = "Additional samples"), linewidth = 1) +
  
  geom_point(data = data[data$Ranking_strategy == "Entropy0-popularity strategy", ], aes(x = Nr_shown_items, y = RMSE), size = 1.5, color = "orange") +
  geom_line(data = data[data$Ranking_strategy == "Entropy0-popularity strategy", ], aes(x = Nr_shown_items, y = RMSE, linetype = "Original"), linewidth = 1, color = "orange") +
  
  theme_light() +
  theme(legend.position = "bottom", legend.title=element_blank()) + 
  labs(x = "Number of shown items to cold user", title = "Entropy0-popularity strategy") +
  scale_x_continuous(breaks = c(10, 25, 50, 100)) +
  scale_linetype_manual(values = "solid", labels = c("Original")) +
  scale_color_manual(values = "grey")

# Save plot as pdf
dev.print(pdf,           # file type
          "Sensitivity_entpop.pdf") # Name of the file

setwd("...")
