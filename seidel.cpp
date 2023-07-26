/*
    Alunos:
    CARLOS HENRIQUE FERNANDES AIRES
    GUILHERME PEREIRA MOURA DA COSTA
    LUIZ FILIPE BRANDI DO NASCIMENTO
*/

#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h>

using namespace std;

#define LINHAS 1000
#define COLUNAS 1000

#define QUATROTs 4
#define OITOTs 8
#define DEZESSEISTs 12

void moduloVetor(double * vetor, int n){
    int i;
    for(i = 0; i < n; i++) vetor[i] = fabsf(vetor[i]);
}

double maiorValor(double * vetor, int n){
    int i;
    double maior = vetor[0];
    for(i = 1; i < n; i++){
        if(vetor[i] > maior) maior = vetor[i];
    }

    return maior;
}

double normaDif(double * xK1, double * xK, int n){
    double dif[n];
    for(int i = 0; i < n; i++){
        dif[i] = xK1[i] - xK[i];
    }
    moduloVetor(dif, n);
    double d = maiorValor(dif, n);
    return d;
}

void seidel(double ** A, double * B, double * xK, int m, int n){
    int iteracao = 1;
    double xK1[n];
    double df;
    
    do{
        memcpy(xK1, xK, n * sizeof(double));
        for(int i = 0; i < m; i++){
            double bi = B[i];
            for(int j = 0; j < n; j++){
                if(j != i){
                    bi -= A[i][j] * xK[j];
                }
            }
            bi /= A[i][i];
 
            // printf("x_%d(%d) = %.6f", i + 1, iteracao, bi);
            // printf("\n");
  
            xK[i] = bi;
        }
        df = normaDif(xK, xK1, n);
        iteracao++;
    }while(df > 10e-6);
    // for(int i = 0; i < COLUNAS; i++) cout << "x_" << i << " = " << xK[i] << "\n";
    // printf("Iterações: %d\n", iteracao);
}

void seidelParalelo(double ** A, double * B, double * xKNovo, int m, int n, int numThreads){
    int iteracao = 1;
    double xKAntigo[n];
    double diffIteracao;
    double bi;
    
    do{
        memcpy(xKAntigo, xKNovo, n * sizeof(double));
        #pragma omp parallel for num_threads(numThreads) schedule(static, n/numThreads) reduction(+:bi) reduction(max:diffIteracao)
        for(int i = 0; i < m; i++){
            bi = B[i];
            for(int j = 0; j < n; j++){
                if(j != i){
                    bi -= A[i][j] * xKNovo[j];
                }
            }
            bi /= A[i][i];

            xKNovo[i] = bi;
        }
        diffIteracao = normaDif(xKNovo, xKAntigo, n);
        iteracao++;
    }while(diffIteracao > 10e-6);
    // for(int i = 0; i < COLUNAS; i++) cout << "x_" << i << " = " << xKNovo[i] << "\n";
    // printf("Iterações: %d\n", iteracao);
}

double ** geraMatriz(int linha, int coluna) {
    double **matriz = new double*[linha];
    int i;

    for (i = 0; i < linha; i++) {
        matriz[i] = new double[coluna];
    }

    matriz[0][0] = -2.02;
    matriz[0][1] = 1.0;

    for (i = 1; i < linha - 1; i++) {
        matriz[i][i - 1] = 1.0;
        matriz[i][i] = -2.02;
        matriz[i][i + 1] = 1.0;
    }

    matriz[i][i - 1] = 1.0;
    matriz[i][i] = -2.02;

    return matriz;
}

double * geraTermosIndependentes(int n) {
    int i = 0;
    double *vetor = new double[n];
    vetor[i] = 1.0;

    for (i = 1; i < n - 1; i++) {
        vetor[i] = 0.0;
    }

    vetor[i] = 1.0;

    return vetor;
}

double *geraX0(int n) {
    double *x0 = new double[n];
    for (int i = 0; i < n; i++) {
        x0[i] = 0.0;
    }

    return x0;
}

void imprimeMatriz(double ** matriz, double * vetor, int linha, int coluna){
    int i, j;
    for(i = 0; i < linha; i++){
        for(j = 0; j < coluna; j++){
            printf("%6.2f", matriz[i][j]);
        }
        printf(" = %6.2f", vetor[i]);
        printf("\n");
    }
}


int main(void){
    double ** matriz = geraMatriz(LINHAS, COLUNAS);    
    double * vetorBase = geraTermosIndependentes(LINHAS);
    double * x0Seidel = geraX0(LINHAS);

    double ** matrizThread = geraMatriz(LINHAS, COLUNAS);    
    double * vetorBaseThread = geraTermosIndependentes(LINHAS);
    double * x0SeidelThread = geraX0(LINHAS);

    printf("\n Gauss Seidel: \n");

    printf("\t A. Sem Threads: \n");
    double time = omp_get_wtime();
    seidel(matriz, vetorBase, x0Seidel, LINHAS, COLUNAS);
    time = omp_get_wtime() - time;
    
    int numThreads = 4;

    printf("\t B. Com Threads: \n");
    double timeThread = omp_get_wtime();
    seidelParalelo(matrizThread, vetorBaseThread, x0SeidelThread, LINHAS, COLUNAS, numThreads);
    timeThread = omp_get_wtime() - timeThread;

    double aumento = timeThread/time * 100;
    printf("\n\t Tempo da execução sem threads: %0.30f\n\n", time);
    printf("\n\t Tempo da execução em paralelo(%d THREADS): %0.30f\n\n", numThreads, timeThread);  
    printf("\n\t Aumento: %0.30f\n\n", aumento);   
}
