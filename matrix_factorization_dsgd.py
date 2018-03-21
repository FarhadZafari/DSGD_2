from __future__ import print_function
from pyspark import SparkContext
import sys
from scipy.sparse import csr_matrix
from scipy import sparse
import numpy as np
from numpy.random import rand
from numpy import savetxt
import random
import logging
import logger
import time

# all comments are why? not what?

class Distributed_Stochastic_Gradient_Decent:
    def __init__(self, num_iterations, num_workers, num_factors, learning_rate, reg):
        self.I = num_iterations
        self.B = num_workers
        self.F = num_factors
        self.beta = learning_rate
        self.lambda_input = reg
        self.user_factors = {}
        self.item_factors = {}

        self.users = set()
        self.items = set()

        self.users_id_to_index = {}
        self.users_index_to_id = {}

        self.items_id_to_index = {}
        self.items_index_to_id = {}

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
            user = line_array[1]
            self.users.add(user)
            user_id = len(self.users) - 1
            if user not in self.users_id_to_index.keys():
                self.users_index_to_id[user_id] = user
                self.users_id_to_index[user] = user_id
            row_indices.append(user_id)
            item = line_array[0]
            self.items.add(item)
            item_id = len(self.items) - 1
            if item not in self.items_id_to_index.keys():
                self.items_index_to_id[item_id] = item
                self.items_id_to_index[item] = item_id
            col_indices.append(item_id)
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
    def L2_loss(self, V, Q, P):
        # temporaray Q and P to calculate the difference
        V_temp = Q.dot(P)
        # number of non zero index
        nz_index = V.nonzero()
        # calcualte the difference after calulating list into array
        difference = np.asarray(V[nz_index] - V_temp[nz_index])
        # calculate the sum
        sum = np.sum(difference ** 2)
        return sum

    def ReadCSV(self, sc, path):
        # input file
        netflix_file = sc.textFile(path)
        # data in tuple form
        ratings_data = netflix_file.map(lambda x: [str(y) for y in x.split(',')])
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        ratings_data_list = ratings_data.collect()

        temp_file = open('tempfile.txt', 'w')
        ratings_data_new_list = []
        for l in ratings_data_list:
            ##################################
            user = l[0]
            self.users.add(user)
            user_id = len(self.users) - 1
            if user not in self.users_id_to_index.keys():
                self.users_id_to_index[user] = user_id
                self.users_index_to_id[user_id] = user
            ##################################
            item = l[1]
            self.items.add(item)
            item_id = len(self.items) - 1
            if item not in self.items_id_to_index.keys():
                self.items_id_to_index[item] = item_id
                self.items_index_to_id[item_id] = item
            ##################################
            rating = int(l[2])
            #The program is expecting to see user ids in the first column, and the item ids in the second column.
            #Since our data puts user ids at first column, I invert the data so that it is compatible with the program.
            temp_file.write(str(item_id) + ',' + str(user_id) + ',' + str(rating) + '\n')
        ratings_data_new_list_back_to_rdd = sc.textFile('tempfile.txt').map(lambda x: [int(y) for y in x.split(',')])
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        return ratings_data_new_list_back_to_rdd

    def train(self):
        start = time.time()
        # made the spark context
        sc = SparkContext(appName="Matric matrix_factorization_dsgd.py using Distributed Stochastic Gradient Descent")

        ratings_data = self.ReadCSV(sc, "ratings.csv")
        # to calculate l2 loss
        self.Nj = ratings_data.keyBy(lambda x: x[0]).countByKey()
        self.Ni = ratings_data.keyBy(lambda x: x[1]).countByKey()

        # here its 2000,2000
        num_items, num_users = ratings_data.reduce(lambda x,y: (max(x[0],y[0]),max(x[1],y[1])))
        num_items = num_items + 1
        num_users = num_users + 1

        print("num_users: ", num_users)
        print("num_items: ", num_items)

        # global varibale to keep track of all previous itertaions
        self.total_update_count = 0

        # initilized Q and P with same number of values as of number of users and items
        # randomizing according to factors provided by the user
        Q = sc.parallelize(range(num_items + 1)).map(lambda x: (x, rand(self.F)))
        P = sc.parallelize(range(num_users + 1)).map(lambda x: (x, rand(self.F)))

        # print_rdd(W)
        # to initialize number of iterations
        iterations = 0
        # algorithm 2
        # run till number of itertaions provided by user
        while iterations < self.I:
            # random number to select startum
            self.random_num = random.randrange(999999)

            # get blocks of parameters Qib and Pib
            Qib = Q.keyBy(lambda x: (hash(str(x[0]) + str(self.random_num)) % self.B))
            Pib = P.keyBy(lambda x: (hash(str(x[0]) + str(self.random_num)) % self.B))

            # get diagonal blocks
            V_diag = ratings_data.filter(lambda x:
                                        ((hash(str(x[0]) + str(self.random_num)) % self.B) == (hash(str(x[1]) + str(self.random_num)) % self.B)))

            # get blocks of data matrices
            V_blocks = V_diag.keyBy(lambda x: (hash(str(x[0]) + str(self.random_num)) % self.B))


            # to keep track of total numbe rof SGD updates made across all strata
            curr_upd_count = V_diag.count()


            # group Vblock, Qib and Pib to send it to SGD update
            V_group = V_blocks.groupWith(Qib, Pib).coalesce(self.B)

            # for x in V_group.collect():
            #     print (x)
            # get the updated Q and P after SGD update
            new_QP = V_group.map(self.SGD_update)

            # separated Q and P
            Q = new_QP.flatMap(lambda x: x[0])
            P = new_QP.flatMap(lambda x: x[1])

            # updated Q and P which are sequence of arrays to form one single array
            Q_Result_temp = np.vstack(Q.sortByKey().map(lambda x: x[1]).collect()[1:])
            P_Result_temp = np.vstack(P.sortByKey().map(lambda x: x[1]).collect()[1:])
            # transpose to multiple Q and P
            Q_Result = Q_Result_temp
            P_Result = P_Result_temp.T

            # update total updates or 'n' in algorithm 2 after each iteration
            self.total_update_count += curr_upd_count
            # print the l2 loss for alogrithm
            # send sparse matrix, Q and P to find the loss
            # print (L2_loss(CSV_to_sparse(netflix_file), Q_Result_temp, P_Result))
            # increment the loop
            iterations += 1

        # L2 Loss after number of iterations
        #print ("Loss is =======================>" + self.L2_loss(self.CSV_to_sparse(netflix_file), Q_Result_temp, P_Result))
        # M = sc.parallelize(Q_Result_temp.dot(P_Result))
        savetxt("Q.txt", Q_Result_temp, delimiter=',')
        savetxt("P.txt", P_Result_temp, delimiter=',')
        # M.coalesce(1, True).saveAsTextFile("output")

        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        self.exportMatrices(Q_Result_temp, P_Result_temp)
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        # run your code
        end = time.time()
        elapsed = end - start
        print("======================================== Elapsed time is: ", elapsed)

    def exportMatrices(self, Q_Result, P_Result):
        print("self.users_id_to_index", self.users_id_to_index)
        print("self.users_index_to_id", self.users_index_to_id)
        print("self.items_id_to_index", self.items_id_to_index)
        print("self.items_index_to_id", self.items_index_to_id)
        print('$$$$$$$$$$$$$$$$$$$$$$$$$exportMatrices$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        user_id = 0
        print("P_Result (users):", P_Result)
        print("Q_Result (items):", Q_Result)

        for l in P_Result:
            self.user_factors[self.users_index_to_id[user_id]] = l
            user_id = user_id + 1

        item_id = 0
        for l in Q_Result:
            self.item_factors[self.items_index_to_id[item_id]] = l
            item_id = item_id + 1

        print(len(self.user_factors))
        print(len(self.item_factors))
        print('$$$$$$$$$$$$$$$$$$$$$$$$$exportMatrices$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')


