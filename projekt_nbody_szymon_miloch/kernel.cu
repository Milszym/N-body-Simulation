//Szymon Miloch 154921 Projekt CUDA
// N-Body, czyli w skrócie symulacja przyciągania cząsteczek pod wpływem grawitacji
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <GL/glut.h>
#include <stdio.h>
#include <stdint.h>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <math.h>

#define TILE_DIM 32
#define BLOCK_ROWS 8

using namespace std;

//ZMIENNE GLOBALNE np pozycje czy ilosc wszystkich 'particles'
GLfloat *posX, *posY, *posZ, *velX, *velY, *velZ, *mass, *radius, *newX, *newY, *newZ;
uint64_t numberOfParticles;
int choice, choice2;

// wpółrzędne położenia obserwatora

GLdouble eyex = -2.0;
GLdouble eyey = -2.0;
GLdouble eyez = -3.0;

// współrzędne punktu w którego kierunku jest zwrócony obserwator,

GLdouble centerx = 0;
GLdouble centery = 0;
GLdouble centerz = 0;

static int ortho = 0;

cudaError_t addWithCuda(uint64_t q, GLfloat *x, GLfloat *y, GLfloat *z, GLfloat *m, GLfloat *vx, GLfloat *vy, GLfloat *vz, GLfloat *nx, GLfloat *ny, GLfloat *nz);

void Timer(int sth) {
	glutPostRedisplay();
	glutTimerFunc(25, Timer, 0);
}

void Draw();
//void Initialize();
void reshape(int w, int h);
void _check_gl_error();
//obsluga klawiatury
void SpecialKeys(int key, int x, int y);
void Keyboard(unsigned char key, int x, int y);




__global__ void nextStep(uint64_t *quantity, GLfloat *x, GLfloat *y, GLfloat *z, GLfloat *m, GLfloat *vx, GLfloat *vy, GLfloat *vz, GLfloat *nextX, GLfloat *nextY, GLfloat *nextZ)
{
	GLfloat deltaX, deltaY, deltaZ, r, deltaT, ax, ay, az;
	//'timestep' później dobiorę wartość optymalną
	deltaT = 0.001;
	GLfloat epsilon = 0.0001;
	//inicjacja wektorów w których zostanie zapisany kolejny krok każdej z cząsteczek

	int i = blockIdx.x * blockDim.x + threadIdx.x;

		//za każdym razem zerujemy przyspieszenia, bo obliczamy przyspieszenie w danym czasie względem reszty cząsteczek
		ax = 0; ay = 0; az = 0;
		//nextX[i] = 0; nextY[i] = 0; nextZ[i] = 0;
		//for (int j = 0; j < *quantity; j++) {
		for (int j=i; j < *quantity; j++) {
			if (i != j){
				deltaX = x[j] - x[i];
				deltaY = y[j] - y[i];
				deltaZ = z[j] - z[i];
				//obliczanie odległości ze wzoru, a przy okazji odwrócenie jej na potrzeby kolejnego wzoru
				//r = 1 / (GLfloat)(sqrtl((float)deltaX*(float)deltaX + (float)deltaY*(float)deltaY + (float)deltaZ*(float)deltaZ));
				r = 1 / (norm3df(deltaX, deltaY, (deltaZ + epsilon)));
				//obliczenie przyspieszenia ze wzoru na sile grawitacji
				ax += m[j] * r*r*r * deltaX;
				ay += m[j] * r*r*r * deltaY;
				az += m[j] * r*r*r * deltaZ;

			}
		}
		//kolejne polozenie poszczegolnych czasteczek, bedzie to suma: poprzedniej lokalizacji, poprzedniej predkości przemnożonej przez 'timestep' 
		//oraz wartości przyspieszenia przemnożonej przez timestep do kwadratu oraz przemnożonej przez 0.5, co wynika z obliczenia całki
		nextX[i] = x[i] + vx[i] * deltaT + 0.5*deltaT*deltaT*ax;
		nextY[i] = y[i] + vy[i] * deltaT + 0.5*deltaT*deltaT*ay;
		nextZ[i] = z[i] + vz[i] * deltaT + 0.5*deltaT*deltaT*az;
		//uaktualniamy prędkość 'poprzednią' (polozenia jeszcze nie, aby nie kolidowalo w kolejnych obliczeniach),
		//gdzie przyspieszenie jest to pochodna predkosci po czasie, a wiec predkosc wynosi:
		vx[i] += deltaT*ax;
		vy[i] += deltaT*ay;
		vz[i] += deltaT*az;
	
	//po wykonaniu obliczen interakcji kazdej czasteczki z kazda, mozemy uaktualnić sytuacje w naszym wszechswiecie
	//osobna pętla jest po to, by kolejne cząsteczki nie brały nowej wartosci polozenia czasteczki, tylko ta w danym 'timestepie'

		//przemieszczenie
		x[i] = nextX[i];
		y[i] = nextY[i];
		z[i] = nextZ[i];

	

}

int main(int iArgc, char** cppArgv)
{
	cout << "Prosze podac liczbe czasteczek:" << endl;
	cin >> numberOfParticles;

	srand(time(NULL));

	posX = new GLfloat[numberOfParticles];
	posY = new GLfloat[numberOfParticles];
	posZ = new GLfloat[numberOfParticles];
	velX = new GLfloat[numberOfParticles];
	velY = new GLfloat[numberOfParticles];
	velZ = new GLfloat[numberOfParticles];
	mass = new GLfloat[numberOfParticles];
	radius = new GLfloat[numberOfParticles];
	newX = new GLfloat[numberOfParticles];
	newY = new GLfloat[numberOfParticles];
	newZ = new GLfloat[numberOfParticles];


	//d - dokladnosc -> im dam większy mnożnik tym więcej cyfr po przecinku będzie w losowanych pozycjach
	//dla d=1 dokladnosc to 3 cyfry po przecinku
	int d = 1;
	int temporary, tmp2;
	int licznik=0;

	cout << endl << "Prosze wybrac jeden z dostepnych trybow: " << endl;
	cout << "1. Losowe rozmieszczenie czasteczek po calej scenie 3d." << endl;
	cout << "2. Czasteczki rozmieszczone w srodku sceny (mniej rozległy zakres losowania)." << endl;
	cout << "3. Czasteczki rozmieszczone w srodku sceny (jeszcze mniej rozległy zakres losowania)." << endl;
	cout << "4. Czasteczki rozmieszczone w rownych odleglosciach w formie szescianu." << endl;
	//cout << "5. Czasteczki rozmieszczone losowo wraz z niewielka iloscia planet. " << endl;
	cin >> choice;
	cout << endl;

	for (int i = 0; i < numberOfParticles; i++) {
		//losowanie poczatkowych pozycji
		posX[i] = ((float)(rand() % 1900 * d - 949 * d)) / 1000 * d;
		posY[i] = ((float)(rand() % 1900 * d - 949 * d)) / 1000 * d;
		posZ[i] = ((float)(rand() % 1900 * d - 949 * d)) / 100 * d;
		//zerowanie poczatkowych predkosci i nowych punktow (byl problem z alokowaniem pamieci wiec robie to w funkcji main)
		velX[i] = 0.0;
		velY[i] = 0.0;
		velZ[i] = 0.0;
		newX[i] = 0.0;
		newY[i] = 0.0;
		newZ[i] = 0.0;
		//jeszcze nie dobralem odpowiedniego zakresu dla masy
		mass[i] = ((float)((rand() % (2000 * d)) / 100));
		//promien czasteczki zalezny od masy
		radius[i] = mass[i] / 10000;
	}

	if(choice == 2){
		for (int i = 0; i < numberOfParticles; i++) {
			//losowanie poczatkowych pozycji
			posX[i] = ((float)(rand() % 190 * d - 94 * d)) / 100 * d;
			posY[i] = ((float)(rand() % 190 * d - 94 * d)) / 100 * d;
			posZ[i] = ((float)(rand() % 190 * d - 94 * d)) / 100 * d;
			//zerowanie poczatkowych predkosci i nowych punktow (byl problem z alokowaniem pamieci wiec robie to w funkcji main)
			velX[i] = 0.0;
			velY[i] = 0.0;
			velZ[i] = 0.0;
			newX[i] = 0.0;
			newY[i] = 0.0;
			newZ[i] = 0.0;
			//jeszcze nie dobralem odpowiedniego zakresu dla masy
			mass[i] = ((float)((rand() % (2000 * d)) / 100));
			//promien czasteczki zalezny od masy
			radius[i] = mass[i] / 10000;
		}
		eyex = -2.0;
		eyey = -2.0;
		eyez = -5.0;
	}
	else if (choice == 4){

		//eyex = 5.0;
		//eyey = 5.0;
		//eyez = -5.0;

		mass[0] = ((float)((rand() % (2000 * d)) / 100));
		temporary = cbrt(numberOfParticles);
		for (int i = 0; i < numberOfParticles; i++) {
			mass[i] = mass[0];
			radius[i] = mass[i] / 10000;
		}
		for (int i = 0; i < temporary; i++){
			for (int j = 0; j < temporary; j++){
				for (int k = 0; k < temporary; k++){
					//tmp2 = i*j*k+licznik;
						posX[licznik] = (i)*0.1;
						posY[licznik] = (j)*0.1;
						posZ[licznik] = (k)*0.1;
						licznik++;
				}
			}
		}
	}
	else if (choice == 3){
		for (int i = 0; i < numberOfParticles; i++) {
			//losowanie poczatkowych pozycji
			posX[i] = ((float)(rand() % 190 * d - 94 * d)) / 1000 * d;
			posY[i] = ((float)(rand() % 190 * d - 94 * d)) / 1000 * d;
			posZ[i] = ((float)(rand() % 190 * d - 94 * d)) / 1000 * d;
			//zerowanie poczatkowych predkosci i nowych punktow (byl problem z alokowaniem pamieci wiec robie to w funkcji main)
			velX[i] = 0.0;
			velY[i] = 0.0;
			velZ[i] = 0.0;
			newX[i] = 0.0;
			newY[i] = 0.0;
			newZ[i] = 0.0;
			//jeszcze nie dobralem odpowiedniego zakresu dla masy
			mass[i] = ((float)((rand() % (2000 * d)) / 100));
			//promien czasteczki zalezny od masy
			radius[i] = mass[i] / 10000;
		}
		eyex = -2.0;
		eyey = -2.0;
		eyez = -8.0;
	}
	//else if (choice == 5){
	//posX[0] = 0.0; posY[0] = 0.0; posZ[0] = 0.0;
	//mass[0] = 20000;
	//radius[0] = mass[0] / 1000000;
	//}

	glutInit(&iArgc, cppArgv);
	if (iArgc > 1){
		ortho = 1;
	}
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(1000, 1000);
	glutCreateWindow("N-Body Miloszkowe");
	//initialize
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glShadeModel(GL_FLAT);

	// dołączenie funkcji obsługi klawiatury
	glutKeyboardFunc(Keyboard);
	// dołączenie funkcji obsługi klawiszy funkcyjnych i klawiszy kursora
	glutSpecialFunc(SpecialKeys);

	_check_gl_error();
	glutDisplayFunc(Draw);
	glutReshapeFunc(reshape);
	Timer(0);
	//glutPostRedisplay();
	glutMainLoop();



	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(uint64_t *q, GLfloat *x, GLfloat *y, GLfloat *z, GLfloat *m, GLfloat *vx, GLfloat *vy, GLfloat *vz, GLfloat *nx, GLfloat *ny, GLfloat *nz)
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
	uint64_t *dev_q = 0;
	GLfloat *dev_x = 0;
	GLfloat *dev_y = 0;
	GLfloat *dev_z = 0;
	GLfloat *dev_vx = 0;
	GLfloat *dev_vy = 0;
	GLfloat *dev_vz = 0;
	GLfloat *dev_m = 0;
	GLfloat *dev_nextX = 0;
	GLfloat *dev_nextY = 0;
	GLfloat *dev_nextZ = 0;

	int size = *q;

	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .

	cudaStatus = cudaMalloc((void**)&dev_q, sizeof(uint64_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_x, size * sizeof(GLfloat));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_y, size * sizeof(GLfloat));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_z, size * sizeof(GLfloat));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_vx, size * sizeof(GLfloat));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_vy, size * sizeof(GLfloat));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_vz, size * sizeof(GLfloat));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_m, size * sizeof(GLfloat));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_nextX, size * sizeof(GLfloat));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_nextY, size * sizeof(GLfloat));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_nextZ, size * sizeof(GLfloat));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.

	cudaStatus = cudaMemcpy(dev_q, q, sizeof(uint64_t), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_x, x, size*sizeof(GLfloat), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_y, y, size*sizeof(GLfloat), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_z, z, size*sizeof(GLfloat), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_m, m, size*sizeof(GLfloat), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_vx, vx, size*sizeof(GLfloat), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_vy, vy, size*sizeof(GLfloat), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_vz, vz, size*sizeof(GLfloat), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_nextX, nx, size*sizeof(GLfloat), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_nextY, ny, size*sizeof(GLfloat), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_nextZ, nz, size*sizeof(GLfloat), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}


	// Launch a kernel on the GPU with one thread for each element.
	//addKernel << <1, size >> >(dev_c, dev_a, dev_b);
	int THREADS = 32;
	int BLOCKS = (size / THREADS) + 1;

	nextStep << < BLOCKS, THREADS >> >(dev_q, dev_x, dev_y, dev_z, dev_m, dev_vx, dev_vy, dev_vz, dev_nextX, dev_nextY, dev_nextZ);
	//nextStep2 << < 1, 512 >> >(dev_q, dev_x, dev_y, dev_z, dev_m, dev_vx, dev_vy, dev_vz, dev_nextX, dev_nextY, dev_nextZ);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(x, dev_x, size * sizeof(GLfloat), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(y, dev_y, size * sizeof(GLfloat), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(z, dev_z, size * sizeof(GLfloat), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(vx, dev_vx, size * sizeof(GLfloat), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(vy, dev_vy, size * sizeof(GLfloat), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(vz, dev_vz, size * sizeof(GLfloat), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(nx, dev_nextX, size * sizeof(GLfloat), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(ny, dev_nextY, size * sizeof(GLfloat), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(nz, dev_nextZ, size * sizeof(GLfloat), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}



Error:
	cudaFree(dev_q);
	cudaFree(dev_x);
	cudaFree(dev_y);
	cudaFree(dev_z);
	cudaFree(dev_vx);
	cudaFree(dev_vy);
	cudaFree(dev_vz);
	cudaFree(dev_m);
	cudaFree(dev_nextX);
	cudaFree(dev_nextY);
	cudaFree(dev_nextZ);

	return cudaStatus;
}

void Draw() {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	//glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	if (ortho) {
	}
	else {
		/* This only rotates and translates the world around to look like the camera moved. */
		//gluLookAt(-2.0, -2.0, -3.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
		gluLookAt(eyex, eyey, eyez, centerx, centery, centerz, 0, 1, 0);
	}
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	int kolor;
	float rozmiar = 1;
	float kolor1, kolor2, kolor3;
	for (int i = 0; i < numberOfParticles; i++) {
		rozmiar = radius[i];
		glPushMatrix();
		//pozycja + kolor+ rysowanie sfer
		glTranslatef(posY[i], posX[i], posZ[i]);
		//kolor = rand() % 3 + 1;
		if (radius[i]>0.0015) {
			kolor1 = 0.9;
			kolor2 = 0.3;
			kolor3 = 0.2;
		}
		else if (radius[i] <= 0.0015 && radius[i] >= 0.001) {
			kolor1 = 0.3;
			kolor2 = 0.2;
			kolor3 = 0.9;
		}
		else if (radius[i]<0.001) {
			kolor1 = 0.2;
			kolor2 = 0.9;
			kolor3 = 0.3;
		}
		glColor4f(kolor1, kolor2, kolor3, 0.2);
		glutSolidSphere(rozmiar * 10, 20, 20);

		glColor4f(kolor1, kolor2, kolor3, 0.5);
		glutSolidSphere(rozmiar * 7, 20, 20);

		glColor4f(kolor1, kolor2, kolor3, 0.9);
		glutSolidSphere(rozmiar * 5, 20, 20);

		glColor4f(1.0, 1.0, 1.0, 1.0);
		glutSolidSphere(rozmiar * 2, 20, 20);

		glPopMatrix();

	}

	glFlush();
	glutSwapBuffers();

	//for (int i = 0; i < numberOfParticles; i++){
	//	cout << posX[i] << "  ";
	//}
	//cout << endl << endl << endl;

	uint64_t *n;
	n = new uint64_t();
	*n = numberOfParticles;

	//if (choice == 4){
		posX[0] = 0.0;
		posY[0] = 0.0;
		posZ[0] = 0.0;
	//}

	cudaError_t cudaStatus = addWithCuda(n, posX, posY, posZ, mass, velX, velY, velZ, newX, newY, newZ);
	//Sleep(1000);
	//for (int i = 0; i < numberOfParticles; i++){
	//	cout << posX[i] << "  ";
	//}
	//cout << endl << endl << endl;
}

void reshape(int w, int h) {
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	if (ortho) {
		glOrtho(-2.0, 2.0, -2.0, 2.0, -1.5, 1.5);
	}
	else {
		glFrustum(-1.0, 1.0, -1.0, 1.0, 1.5, 20.0);
	}
	glMatrixMode(GL_MODELVIEW);

}

void _check_gl_error() {
	GLenum err(glGetError());

	while (err != GL_NO_ERROR) {
		string error;

		switch (err) {
		case GL_INVALID_OPERATION:      error = "INVALID_OPERATION";      break;
		case GL_INVALID_ENUM:           error = "INVALID_ENUM";           break;
		case GL_INVALID_VALUE:          error = "INVALID_VALUE";          break;
		case GL_OUT_OF_MEMORY:          error = "OUT_OF_MEMORY";          break;
			//case GL_INVALID_FRAMEBUFFER_OPERATION:  error = "INVALID_FRAMEBUFFER_OPERATION";  break;
		}

		cerr << "GL_" << error.c_str() << " - ";
		err = glGetError();
	}
}


void Keyboard(unsigned char key, int x, int y)
{
	// klawisz +
	if (key == '+')
		eyez -= 0.1;
	else

		// klawisz -
	if (key == '-')
		eyez += 0.1;

	// odrysowanie okna
	//Reshape(glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT));
}

// obsługa klawiszy funkcyjnych i klawiszy kursora

void SpecialKeys(int key, int x, int y)
{
	switch (key)
	{
		// kursor w lewo
	case GLUT_KEY_LEFT:
		eyex += 0.1;
		break;

		// kursor w górę
	case GLUT_KEY_UP:
		eyey -= 0.1;
		break;

		// kursor w prawo
	case GLUT_KEY_RIGHT:
		eyex -= 0.1;
		break;

		// kursor w dół
	case GLUT_KEY_DOWN:
		eyey += 0.1;
		break;
	}

	// odrysowanie okna
	//Reshape(glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT));
}