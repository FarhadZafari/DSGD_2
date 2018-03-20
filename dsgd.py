from click import command, option
import matrix_factorization_dsgd

@command()
@option('--num_iterations', default = 5, help='Number of iterations.')
@option('--num_workers', default = 5, help='Number of workers.')
@option('--num_factors', default = 5, help='Number of factors.')
@option('--learning_rate', default = 0.6, help='Learning rate.')
@option('--reg', default = 0.01, help='Regularization.')
def main(num_iterations, num_workers, num_factors, learning_rate, reg):
    DSGD = matrix_factorization_dsgd.Distributed_Stochastic_Gradient_Decent(
        num_iterations, num_workers, num_factors, learning_rate, reg)
    DSGD.train()

if __name__ == '__main__':
    main()