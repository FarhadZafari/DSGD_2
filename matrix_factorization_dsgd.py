from __future__ import print_function
from pyspark import SparkContext
import sys
from scipy.sparse import csr_matrix
from scipy import sparse
import numpy as np
from numpy.random import rand
from numpy import savetxt
import random
from click import command, option
# all comments are why? not what?


# # number of iterations
# I = int(sys.argv[1])
# # number of workers
# B = int(sys.argv[2])
# # number of factors
# F = int(sys.argv[3])
# # beta value
# beta = float(sys.argv[4])
# # lambda input
# lambda_input = float(sys.argv[5])

# global random_num, total_update_count
# global Nj, Ni

class Distributed_Stochastic_Gradient_Decent:
    def __init__(self, num_iterations, num_workers, num_factors, learning_rate, reg):
        self.I = num_iterations
        self.B = num_workers
        self.F = num_factors
        self.beta = learning_rate
        self.lambda_input = reg
        self.user_factors = {}
        self.item_factors = {}

    # small method to print the contents of RDD
    def print_rdd(self, RDD):
        for line in RDD.collect():
            print (line)

    # to tranfrom each line into a vector
    def create_matrix(self, x):
        # split the comma separated file
        matrix_elements = x.split(',')
        # convert string values to integer and create a matrix of three values(list)
        matrix = [int(matrix_elements[0]), int(matrix_elements[1]), float(matrix_elements[2])]
        # return the tuple
        return matrix

    # changed the data to sparse matrix formation to calculate Reconstruction error
    def CSV_to_sparse(self, file):
        # create different lists
        row_indices = []
        col_indices = []
        data_rating = []

        lines = file.collect()
        for line in lines:
            line_array = line.split(",")
            row_indices.append(int(line_array[0]) - 1)
            col_indices.append(int(line_array[1]) - 1)
            data_rating.append(float(line_array[2]))

        return csr_matrix((data_rating, (row_indices, col_indices)))

    def SGD_update(self, t):
        # get all three items
        V_block, W_block, H_block = t[1]
        # converted for easy manipulation
        W_dict = dict(W_block)
        H_dict = dict(H_block)
        # number of SGD updates for current worker or n'
        iter = 0
        # for each item in each tuple
        for (movie_id, user_id, rating) in V_block:
            # increse n'
            iter += 1
            # SGD stepsize
            epsilon = pow(100 + self.total_update_count + iter, -1 * self.beta)

            Wi = W_dict[movie_id]
            Hj = H_dict[user_id]
            # LNZSL
            loss = -2*(rating - np.dot(Wi,Hj))
            H_dict[user_id]  = Hj - epsilon*(2*self.lambda_input/self.Ni[user_id]*Hj + loss*Wi)
            W_dict[movie_id] = Wi - epsilon*(2*self.lambda_input/self.Nj[movie_id]*Wi + loss*Hj)
        return (W_dict.items(), H_dict.items())

    # l2 loss
    def L2_loss(self, V, W, H):
        # temporaray W and H to calculate the difference
        V_temp = W.dot(H)
        # number of non zero index
        nz_index = V.nonzero()
        # calcualte the difference after calulating list into array
        difference = np.asarray(V[nz_index] - V_temp[nz_index])
        # calculate the sum
        sum = np.sum(difference ** 2)
        return sum

    def train(self):
        # made the spark contest
        sc = SparkContext(appName="Matric matrix_factorization_dsgd.py using Distributed Stochastic Gradient Descent")
        # input file
        netflix_file = sc.textFile("nf_subsample.csv")

        # data in tuple form
        ratings_data = netflix_file.map(lambda x: [int(y) for y in x.split(',')])

        # to calculate l2 loss
        self.Nj = ratings_data.keyBy(lambda x: x[0]).countByKey()
        self.Ni = ratings_data.keyBy(lambda x: x[1]).countByKey()

        # here its 2000,2000
        num_movies, num_users = ratings_data.reduce(lambda x,y: (max(x[0],y[0]),max(x[1],y[1])))

        # global varibale to keep track of all previous itertaions
        self.total_update_count = 0

        # initilized W and H with same number of values as of number of users and movies
        # randomizing according to factors provided by the user
        W = sc.parallelize(range(num_movies + 1)).map(lambda x: (x, rand(self.F)))
        H = sc.parallelize(range(num_users + 1)).map(lambda x: (x, rand(self.F)))

        # print_rdd(W)
        # to initialize number of iterations
        iterations = 0
        # algorithm 2
        # run till number of itertaions provided by user
        while iterations < self.I:
            # random number to select startum
            self.random_num = random.randrange(999999)

            # get blocks of parameters Wib and Hib
            Wib = W.keyBy(lambda x: (hash(str(x[0]) + str(self.random_num)) % self.B))
            Hib = H.keyBy(lambda x: (hash(str(x[0]) + str(self.random_num)) % self.B))

            # get diagonal blocks
            V_diag = ratings_data.filter(lambda x:
                                        ((hash(str(x[0]) + str(self.random_num)) % self.B) == (hash(str(x[1]) + str(self.random_num)) % self.B)))

            # get blocks of data matrices
            V_blocks = V_diag.keyBy(lambda x: (hash(str(x[0]) + str(self.random_num)) % self.B))


            # to keep track of total numbe rof SGD updates made across all strata
            curr_upd_count = V_diag.count()


            # group Vblock, Wib and Hib to send it to SGD update
            V_group = V_blocks.groupWith(Wib, Hib).coalesce(self.B)

            # for x in V_group.collect():
            #     print (x)
            # get the updated W and H after SGD update
            new_WH = V_group.map(self.SGD_update)

            # separated W and H
            W = new_WH.flatMap(lambda x: x[0])
            H = new_WH.flatMap(lambda x: x[1])

            # updated W and H which are sequence of arrays to form one single array
            W_result_temp = np.vstack(W.sortByKey().map(lambda x: x[1]).collect()[1:])
            H_result_temp = np.vstack(H.sortByKey().map(lambda x: x[1]).collect()[1:])
            # transpose to multiple W and H
            W_result = W_result_temp
            H_result = H_result_temp.T

            # update total updates or 'n' in algorithm 2 after each iteration
            self.total_update_count += curr_upd_count
            # print the l2 loss for alogrithm
            # send sparse matrix, W and H to find the loss
            # print (L2_loss(CSV_to_sparse(netflix_file), W_result_temp, H_result))
            # increment the loop
            iterations += 1

        # L2 Loss after number of iterations
        print (self.L2_loss(self.CSV_to_sparse(netflix_file), W_result_temp, H_result))
        # M = sc.parallelize(W_result_temp.dot(H_result))
        savetxt("W.txt", W_result, delimiter=',')
        savetxt("H.txt", H_result, delimiter=',')
        # M.coalesce(1, True).saveAsTextFile("output")