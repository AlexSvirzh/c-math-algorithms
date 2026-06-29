#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <windows.h>

void mat_mul(int m, int n, int k, double *A, double *B, double *C) {
    memset(C, 0, m * k * sizeof(double));
    for (int i = 0; i < m; i++) {
        for (int c = 0; c < n; c++) {
            double A_ic = A[i * n + c];
            for (int j = 0; j < k; j++) {
                C[i * k + j] += A_ic * B[c * k + j];
            }
        }
    }
}

void mat_add(int m, int n, double *A, double *B, double *C) {
    for (int i = 0; i < m * n; i++) C[i] = A[i] + B[i];
}

void mat_sub(int m, int n, double *A, double *B, double *C) {
    for (int i = 0; i < m * n; i++) C[i] = A[i] - B[i];
}

void mat_transpose(int m, int n, double *A, double *AT) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) AT[j * m + i] = A[i * n + j];
}

void mat_copy(int size, double *src, double *dst) {
    memcpy(dst, src, size * sizeof(double));
}

double inner_product(int size, double *A, double *B) {
    double sum = 0.0;
    for (int i = 0; i < size; i++) sum += A[i] * B[i];
    return sum;
}

void apply_lyapunov_op(int N, double *X, double *Ak, double *E, double *AkT, double *ET, double *Out, double *T1, double *T2) {
    mat_mul(N, N, N, AkT, X, T1);
    mat_mul(N, N, N, T1, E, Out);
    mat_mul(N, N, N, ET, X, T1);
    mat_mul(N, N, N, T1, Ak, T2);
    mat_add(N, N, Out, T2, Out);
}

void apply_lyapunov_adjoint(int N, double *Y, double *Ak, double *E, double *AkT, double *ET, double *Out, double *T1, double *T2) {
    mat_mul(N, N, N, Ak, Y, T1);
    mat_mul(N, N, N, T1, ET, Out);
    mat_mul(N, N, N, E, Y, T1);
    mat_mul(N, N, N, T1, AkT, T2);
    mat_add(N, N, Out, T2, Out);
}

void solve_lyapunov_cgne(int N, double *Ak, double *E, double *W, double *X) {
    int size = N * N;
    double *AkT = (double*)malloc(size * sizeof(double));
    double *ET  = (double*)malloc(size * sizeof(double));
    mat_transpose(N, N, Ak, AkT);
    mat_transpose(N, N, E, ET);

    double *R  = (double*)malloc(size * sizeof(double));
    double *Z  = (double*)malloc(size * sizeof(double));
    double *P  = (double*)malloc(size * sizeof(double));
    double *V  = (double*)malloc(size * sizeof(double));
    double *T1 = (double*)malloc(size * sizeof(double));
    double *T2 = (double*)malloc(size * sizeof(double));

    apply_lyapunov_op(N, X, Ak, E, AkT, ET, V, T1, T2); 
    for(int i = 0; i < size; i++) R[i] = -W[i] - V[i];

    apply_lyapunov_adjoint(N, R, Ak, E, AkT, ET, Z, T1, T2);

    for(int i = 0; i < size; i++) P[i] = Z[i];

    double tol = 1e-9;
    int max_iter = 3000; 

    for (int iter = 0; iter < max_iter; iter++) {
        double gamma = inner_product(size, Z, Z);
        if (gamma < 1e-20) break;

        apply_lyapunov_op(N, P, Ak, E, AkT, ET, V, T1, T2);

        double v_dot_v = inner_product(size, V, V);
        if (v_dot_v < 1e-20) break;

        double alpha = gamma / v_dot_v;

        for (int i = 0; i < size; i++) {
            X[i] += alpha * P[i];
            R[i] -= alpha * V[i];
        }

        double norm_R = sqrt(inner_product(size, R, R));
        if (norm_R < tol) break;

        apply_lyapunov_adjoint(N, R, Ak, E, AkT, ET, Z, T1, T2);

        double gamma_new = inner_product(size, Z, Z);
        double beta = gamma_new / gamma;

        for (int i = 0; i < size; i++) {
            P[i] = Z[i] + beta * P[i];
        }
    }

    free(AkT); free(ET); free(R); free(Z); free(P); free(V); free(T1); free(T2);
}

void riccati_newton_kleinman(int N, int M, double *A, double *B, double *E, double *Q, double *X) {
    double *K_k  = (double*)malloc(M * N * sizeof(double));
    double *A_k  = (double*)malloc(N * N * sizeof(double));
    double *W    = (double*)malloc(N * N * sizeof(double));
    double *BT   = (double*)malloc(M * N * sizeof(double));
    double *K_kT = (double*)malloc(N * M * sizeof(double));
    double *T1   = (double*)malloc(M * N * sizeof(double));
    double *T2   = (double*)malloc(N * N * sizeof(double));
    double *prev_X = (double*)malloc(N * N * sizeof(double));

    mat_transpose(N, M, B, BT);
    memset(X, 0, N * N * sizeof(double));

    printf("\n--- Запуск метода Ньютона-Клейнмана ---\n");
    for (int step = 1; step <= 50; step++) {
        mat_copy(N * N, X, prev_X);

        mat_mul(M, N, N, BT, X, T1);
        mat_mul(M, N, N, T1, E, K_k);

        mat_mul(N, M, N, B, K_k, T2);
        mat_sub(N, N, A, T2, A_k);

        mat_transpose(M, N, K_k, K_kT);
        mat_mul(N, M, N, K_kT, K_k, T2);
        mat_add(N, N, Q, T2, W);

        solve_lyapunov_cgne(N, A_k, E, W, X);

        double diff = 0.0, normX = 0.0;
        for(int i = 0; i < N * N; i++) {
            double d = X[i] - prev_X[i];
            diff += d * d;
            normX += X[i] * X[i];
        }
        
        double rel_err = sqrt(diff) / (sqrt(normX) + 1e-12);
        printf("Итерация %2d завершена | Относительная разница dX: %e\n", step, rel_err);
        
        if (rel_err < 1e-7 && step > 1) {
            printf("[+] Метод Ньютона успешно сошелся за %d итераций!\n", step);
            break;
        }
    }

    free(K_k); free(A_k); free(W); free(BT); free(K_kT); free(T1); free(T2); free(prev_X);
}

void get_input_path(const char *prompt, char *buffer, int max_len) {
    printf("%s", prompt);
    fflush(stdout); 
    if (fgets(buffer, max_len, stdin) != NULL) {
        size_t len = strlen(buffer);
        while (len > 0 && (buffer[len - 1] == '\n' || buffer[len - 1] == '\r')) {
            buffer[len - 1] = '\0';
            len--;
        }
    }
}

int read_matrix(const char *filename, double **matrix, int *total_elements) {
    FILE *f = fopen(filename, "r");
    if (!f) return 0;
    
    int capacity = 16;
    *total_elements = 0;
    *matrix = (double *)malloc(capacity * sizeof(double));
    double temp;
    int res;
    
    while ((res = fscanf(f, "%lf", &temp)) == 1) {
        (*matrix)[*total_elements] = temp;
        (*total_elements)++;
        if (*total_elements >= capacity) {
            capacity *= 2;
            *matrix = (double *)realloc(*matrix, capacity * sizeof(double));
        }
    }
    
    fclose(f);

    if (res == 0) {
        free(*matrix);
        *matrix = NULL;
        return -1; 
    }

    if (*total_elements == 0) {
        free(*matrix);
        *matrix = NULL;
        return 0; 
    }

    return 1; 
}

int main() {
    SetConsoleCP(65001); SetConsoleOutputCP(65001);
    
    char pathA[512], pathB[512], pathE[512], pathQ[512];
    double *A = NULL, *B = NULL, *E = NULL, *Q = NULL, *X = NULL;
    int countA, countB, countE, countQ;

    printf("ПРОГРАММА РЕШЕНИЯ УРАВНЕНИЯ РИККАТИ (CARE)\n\n");

    get_input_path("Введите путь к файлу матрицы A: ", pathA, sizeof(pathA));
    get_input_path("Введите путь к файлу матрицы B: ", pathB, sizeof(pathB));
    get_input_path("Введите путь к файлу матрицы E: ", pathE, sizeof(pathE));
    get_input_path("Введите путь к файлу матрицы Q: ", pathQ, sizeof(pathQ));

    printf("\nЧтение файлов...\n");

    int status;
    status = read_matrix(pathA, &A, &countA);
    if (status == 0) { printf("[!] Ошибка: файл '%s' не найден или пуст.\n", pathA); return 1; }
    if (status == -1) { printf("[!] Ошибка: в файле '%s' обнаружены недопустимые символы (буквы).\n", pathA); return 1; }

    status = read_matrix(pathB, &B, &countB);
    if (status == 0) { printf("[!] Ошибка: файл '%s' не найден или пуст.\n", pathB); return 1; }
    if (status == -1) { printf("[!] Ошибка: в файле '%s' обнаружены недопустимые символы.\n", pathB); return 1; }

    status = read_matrix(pathE, &E, &countE);
    if (status == 0) { printf("[!] Ошибка: файл '%s' не найден или пуст.\n", pathE); return 1; }
    if (status == -1) { printf("[!] Ошибка: в файле '%s' обнаружены недопустимые символы.\n", pathE); return 1; }

    status = read_matrix(pathQ, &Q, &countQ);
    if (status == 0) { printf("[!] Ошибка: файл '%s' не найден или пуст.\n", pathQ); return 1; }
    if (status == -1) { printf("[!] Ошибка: в файле '%s' обнаружены недопустимые символы.\n", pathQ); return 1; }

    int N = (int)round(sqrt(countA));
    if (N * N != countA || N == 0) {
        printf("[!] Ошибка геометрии: матрица A не является строго квадратной (считано %d элементов).\n", countA);
        return 1;
    }
    
    int M = countB / N;
    if (countB % N != 0 || countB == 0) {
        printf("[!] Ошибка геометрии: размерность B (%d) несовместима со строками системы N (%d).\n", countB, N);
        return 1;
    }

    if (countE != countA) {
        printf("[!] Ошибка геометрии: матрица E должна быть того же размера, что и A (найдено %d).\n", countE);
        return 1;
    }

    if (countQ != countA) {
        printf("[!] Ошибка геометрии: матрица Q должна быть того же размера, что и A (найдено %d).\n", countQ);
        return 1;
    }

    printf("[+] Матрицы успешно загружены и проверены. Размерность системы N=%d, входов M=%d\n", N, M);

    X = (double*)malloc(N * N * sizeof(double));

    riccati_newton_kleinman(N, M, A, B, E, Q, X);

    FILE *fout = fopen("output.txt", "w");
    if (fout) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                fprintf(fout, "%8.4f ", X[i * N + j]);
            }
            fprintf(fout, "\n");
        }
        fclose(fout);
        printf("\n[+] Матрица решения X успешно сохранена в файл 'output.txt'\n");
    } else {
        printf("\n[!] Ошибка: не удалось создать файл 'output.txt' для записи.\n");
    }

    if (N <= 10) {
        printf("\nРЕЗУЛЬТАТ (Стабилизирующее решение X):\n");
        for(int i = 0; i < N; i++) {
            for(int j = 0; j < N; j++) {
                printf("%8.4f ", X[i * N + j]);
            }
            printf("\n");
        }
    } else {
        printf("\n[i] Размерность матрицы (N=%d) больше 10. Детальный вывод в терминал пропущен.\n", N);
    }

    free(A); free(B); free(E); free(Q); free(X);
    
    printf("\nНажмите Enter для завершения...");
    getchar(); 
    return 0;
}
