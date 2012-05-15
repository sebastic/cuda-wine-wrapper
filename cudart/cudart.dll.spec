#
# File: cudart.dll.spec
#
# Copyrighted by Seth Shelnutt under the LGPL v2.1 or later
#
# Wine spec file for the cudart.dll built-in library (a minimal wrapper around the
# linux library libcuart)
#
# For further details of wine spec files see the Winelib documentation at
# www.winehq.org

@  stdcall cudaGetDeviceCount( ptr ) wine_cudaGetDeviceCount
@  stdcall cudaSetDevice( long ) wine_cudaSetDevice
@  stdcall cudaGetDevice( ptr ) wine_cudaGetDevice
@  stdcall cudaGetDeviceProperties( ptr long ) wine_cudaGetDeviceProperties
@  stdcall cudaChooseDevice( ptr  ptr ) wine_cudaChooseDevice
@  stdcall cudaThreadSynchronize(  ) wine_cudaThreadSynchronize
@  stdcall cudaThreadExit(  ) wine_cudaThreadExit
@  stdcall cudaStreamCreate( ptr ) wine_cudaStreamCreate
@  stdcall cudaStreamQuery( long ) wine_cudaStreamQuery
@  stdcall cudaStreamSynchronize( long ) wine_cudaStreamSynchronize
@  stdcall cudaStreamDestroy( long ) wine_cudaStreamDestroy
@  stdcall cudaEventCreate( ptr ) wine_cudaEventCreate
@  stdcall cudaEventRecord( long long ) wine_cudaEventRecord
@  stdcall cudaEventQuery( long ) wine_cudaEventQuery
@  stdcall cudaEventSynchronize( long ) wine_cudaEventSynchronize
@  stdcall cudaEventDestroy( long ) wine_cudaEventDestroy
@  stdcall cudaEventElapsedTime( ptr long long ) wine_cudaEventElapsedTime
@  stdcall cudaConfigureCall( long long long long) wine_cudaConfigureCall
@  stdcall cudaLaunch( ptr ) wine_cudaLaunch
@  stdcall cudaSetupArgument( ptr long long ) wine_cudaSetupArgument
@  stdcall cudaMalloc( ptr long ) wine_cudaMalloc
@  stdcall cudaMallocPitch( ptr ptr long long ) wine_cudaMallocPitch
@  stdcall cudaFree( ptr ) wine_cudaFree
@  stdcall cudaMallocArray( ptr ptr long long ) wine_cudaMallocArray
@  stdcall cudaFreeArray( ptr ) wine_cudaFreeArray
@  stdcall cudaMallocHost( ptr long ) wine_cudaMallocHost
@  stdcall cudaFreeHost( ptr ) wine_cudaFreeHost
@  stdcall cudaMemset( ptr long long ) wine_cudaMemset
@  stdcall cudaMemset2D( ptr long long long long ) wine_cudaMemset2D
@  stdcall cudaMemcpy( ptr ptr long  long ) wine_cudaMemcpy
@  stdcall cudaMemcpyAsync( ptr ptr long  long long ) wine_cudaMemcpyAsync
@  stdcall cudaMemcpy2D( ptr long ptr long long long long ) wine_cudaMemcpy2D
@  stdcall cudaMemcpy2DAsync( ptr long ptr long long long long long ) wine_cudaMemcpy2DAsync
@  stdcall cudaMemcpyToArray( ptr long long ptr long long ) wine_cudaMemcpyToArray
@  stdcall cudaMemcpyToArrayAsync( ptr long long ptr long long long ) wine_cudaMemcpyToArrayAsync
@  stdcall cudaMemcpy2DToArray( ptr long long ptr long long long long ) wine_cudaMemcpy2DToArray
@  stdcall cudaMemcpy2DToArrayAsync( ptr long long ptr long long long long long ) wine_cudaMemcpy2DToArrayAsync
@  stdcall cudaMemcpyFromArray( ptr ptr long long long long ) wine_cudaMemcpyFromArray
@  stdcall cudaMemcpyFromArrayAsync( ptr ptr long long long long long ) wine_cudaMemcpyFromArrayAsync
@  stdcall cudaMemcpy2DFromArray( ptr long ptr long long long long long ) wine_cudaMemcpy2DFromArray
@  stdcall cudaMemcpy2DFromArrayAsync( ptr long ptr long long long long long long ) wine_cudaMemcpy2DFromArrayAsync
@  stdcall cudaMemcpyArrayToArray( ptr long long ptr long long long long ) wine_cudaMemcpyArrayToArray
@  stdcall cudaMemcpy2DArrayToArray( ptr long long ptr long long long long long ) wine_cudaMemcpy2DArrayToArray
@  stdcall cudaMemcpyToSymbol( ptr ptr long long long ) wine_cudaMemcpyToSymbol
@  stdcall cudaMemcpyFromSymbol( ptr long long long long ) wine_cudaMemcpyFromSymbol
@  stdcall cudaGetSymbolAddress( ptr ptr ) wine_cudaGetSymbolAddress
@  stdcall cudaGetSymbolSize( ptr ptr ) wine_cudaGetSymbolSize

@  stdcall cudaMalloc3D( ptr long ) wine_cudaMalloc3D

@  stdcall cudaMalloc3DArray( ptr ptr long ) wine_cudaMalloc3DArray
@  stdcall cudaMemset3D( long long long ) wine_cudaMemset3D

@  stdcall cudaMemcpy3D( long ) wine_cudaMemcpy3D
@  stdcall cudaMemcpy3DAsync( long long ) wine_cudaMemcpy3DAsync
@  stdcall cudaBindTexture( ptr ptr ptr ptr long ) wine_cudaBindTexture
@  stdcall cudaBindTextureToArray( ptr ptr ptr ) wine_cudaBindTextureToArray
@  stdcall cudaUnbindTexture( ptr ) wine_cudaUnbindTexture

@  stdcall cudaGetChannelDesc( ptr ptr ) wine_cudaGetChannelDesc
@  stdcall cudaGetTextureReference( ptr long) wine_cudaGetTextureReference
@  stdcall cudaGetTextureAlignmentOffset( ptr ptr ) wine_cudaGetTextureAlignmentOffset

@  stdcall cudaGLSetGLDevice( long ) wine_cudaGLSetGLDevice
@  stdcall cudaGLRegisterBufferObject( long ) wine_cudaGLRegisterBufferObject
@  stdcall cudaGLMapBufferObject( ptr long ) wine_cudaGLMapBufferObject
@  stdcall cudaGLUnmapBufferObject( long ) wine_cudaGLUnmapBufferObject
@  stdcall cudaGLUnregisterBufferObject( long ) wine_cudaGLUnregisterBufferObject

#Direct3D functions were skipped, implementation at a later time

@  stdcall cudaGetLastError(  ) wine_cudaGetLastError
@  stdcall cudaGetErrorString( long  ) wine_cudaGetErrorString



@ stdcall __cudaRegisterFatBinary( ptr ) wine_cudaRegisterFatBinary
@ stdcall __cudaRegisterFunction( ptr ptr ptr ptr long ptr ptr ptr ptr ptr ) wine_cudaRegisterFunction
@ stdcall __cudaRegisterVar( ptr ptr ptr ptr long long long long ) wine_cudaRegisterVar
@ stdcall __cudaRegisterShared( ptr ptr ) wine_cudaRegisterShared
@ stdcall __cudaRegisterSharedVar( ptr ptr long long long ) wine_cudaRegisterSharedVar
@ stdcall __cudaUnregisterFatBinary( ptr ) wine_cudaUnregisterFatBinary


#New to cuda 3.0
@ stdcall __cudaMutexOperation( long ) wine_cudaMutexOperation
@ stdcall __cudaRegisterTexture( ptr ptr ptr ptr long long long ) wine_cudaRegisterTexture
@ stdcall __cudaSynchronizeThreads( ) wine_cudaSynchronizeThreads
@ stdcall __cudaTextureFetch( ptr ptr long ptr) wine_cudaTextureFetch
@ stdcall cudaBindTexture2D( ptr ptr ptr ptr long long long ) wine_cudaBindTexture2D
@ stdcall cudaCreateChannelDesc( long long long long long ) wine_cudaCreateChannelDesc
@ stdcall cudaDriverGetVersion( ptr ) wine_cudaDriverGetVersion
@ stdcall cudaEventCreateWithFlags( ptr long) wine_cudaEventCreateWithFlags
@ stdcall cudaFuncGetAttributes( ptr ptr ) wine_cudaFuncGetAttributes
@ stdcall cudaFuncSetCacheConfig( ptr long ) wine_cudaFuncSetCacheConfig
@ stdcall cudaGLMapBufferObjectAsync( ptr long long ) wine_cudaGLMapBufferObjectAsync
@ stdcall cudaGLSetBufferObjectMapFlags( long long ) wine_cudaGLSetBufferObjectMapFlags
@ stdcall cudaGLUnmapBufferObjectAsync( long long ) wine_cudaGLUnmapBufferObjectAsync
@ stdcall cudaGraphicsGLRegisterBuffer( ptr long long ) wine_cudaGraphicsGLRegisterBuffer
@ stdcall cudaGraphicsGLRegisterImage( ptr long long long ) wine_cudaGraphicsGLRegisterImage
@ stdcall cudaGraphicsMapResources( long ptr long ) wine_cudaGraphicsMapResources
@ stdcall cudaGraphicsResourceGetMappedPointer( ptr ptr ptr ) wine_cudaGraphicsResourceGetMappedPointer
@ stdcall cudaGraphicsResourceSetMapFlags( ptr long ) wine_cudaGraphicsResourceSetMapFlags
@ stdcall cudaGraphicsSubResourceGetMappedArray( ptr ptr long long ) wine_cudaGraphicsSubResourceGetMappedArray
@ stdcall cudaGraphicsUnmapResources( long ptr long ) wine_cudaGraphicsUnmapResources
@ stdcall cudaGraphicsUnregisterResource( ptr ) wine_cudaGraphicsUnregisterResource
@ stdcall cudaHostAlloc( ptr long long ) wine_cudaHostAlloc
@ stdcall cudaHostGetDevicePointer( ptr ptr long ) wine_cudaHostGetDevicePointer
@ stdcall cudaHostGetFlags( ptr ptr ) wine_cudaHostGetFlags
@ stdcall cudaMemGetInfo( ptr ptr ) wine_cudaMemGetInfo
@ stdcall cudaMemcpyFromSymbolAsync( ptr ptr long long long long ) wine_cudaMemcpyFromSymbolAsync
@ stdcall cudaMemcpyToSymbolAsync( ptr ptr long long long long ) wine_cudaMemcpyToSymbolAsync
@ stdcall cudaRuntimeGetVersion( ptr ) wine_cudaRuntimeGetVersion
@ stdcall cudaSetDeviceFlags( long ) wine_cudaSetDeviceFlags
@ stdcall cudaSetDoubleForDevice( ptr ) wine_cudaSetDoubleForDevice
@ stdcall cudaSetDoubleForHost( ptr ) wine_cudaSetDoubleForHost
@ stdcall cudaSetValidDevices( ptr long ) wine_cudaSetValidDevices
#@ stdcall cudaWGLGetDevice( ptr long ) wine_cudaWGLGetDevice

#New to cuda 4.2
@  stdcall cudaDeviceReset(  ) wine_cudaDeviceReset
@  stdcall cudaDeviceSynchronize(  ) wine_cudaDeviceSynchronize
@  stdcall cudaDeviceSetLimit( long long ) wine_cudaDeviceSetLimit
@  stdcall cudaDeviceGetLimit( ptr long ) wine_cudaDeviceGetLimit
@  stdcall cudaDeviceGetCacheConfig( ptr ) wine_cudaDeviceGetCacheConfig
@  stdcall cudaDeviceSetCacheConfig( long ) wine_cudaDeviceSetCacheConfig
@  stdcall cudaDeviceGetSharedMemConfig( ptr ) wine_cudaDeviceGetSharedMemConfig
@  stdcall cudaDeviceSetSharedMemConfig( long ) wine_cudaDeviceSetSharedMemConfig
@  stdcall cudaDeviceGetByPCIBusId( ptr ptr ) wine_cudaDeviceGetByPCIBusId
@  stdcall cudaDeviceGetPCIBusId( ptr long long ) wine_cudaDeviceGetPCIBusId
@  stdcall cudaIpcOpenEventHandle( ptr long ) wine_cudaIpcOpenEventHandle
@  stdcall cudaIpcGetMemHandle( ptr ptr ) wine_cudaIpcGetMemHandle
@  stdcall cudaIpcOpenMemHandle( ptr long long ) wine_cudaIpcOpenMemHandle
@  stdcall cudaIpcCloseMemHandle( ptr ) wine_cudaIpcCloseMemHandle
@  stdcall cudaThreadSetLimit( long long ) wine_cudaThreadSetLimit
@  stdcall cudaThreadGetLimit( ptr long ) wine_cudaThreadGetLimit
@  stdcall cudaThreadGetCacheConfig( ptr ) wine_cudaThreadGetCacheConfig
@  stdcall cudaThreadSetCacheConfig( long ) wine_cudaThreadSetCacheConfig
@  stdcall cudaPeekAtLastError(  ) wine_cudaPeekAtLastError
@  stdcall cudaStreamWaitEvent( long long long ) wine_cudaStreamWaitEvent
@  stdcall cudaFuncSetSharedMemConfig( ptr long ) wine_cudaFuncSetSharedMemConfig
@  stdcall cudaHostRegister( ptr long long ) wine_cudaHostRegister
@  stdcall cudaHostUnregister( ptr ) wine_cudaHostUnregister
@  stdcall cudaHostGetDevicePointer( ptr ptr long ) wine_cudaHostGetDevicePointer
@  stdcall cudaMemcpy3DPeer( ptr ) wine_cudaMemcpy3DPeer
@  stdcall cudaMemcpy3DPeerAsync( ptr long ) wine_cudaMemcpy3DPeerAsync
@  stdcall cudaArrayGetInfo( ptr ptr ptr ptr ) wine_cudaArrayGetInfo
@  stdcall cudaMemcpyPeer( ptr long ptr long long ) wine_cudaMemcpyPeer
@  stdcall cudaMemcpyPeerAsync( ptr long ptr long long long ) wine_cudaMemcpyPeerAsync
@  stdcall cudaMemsetAsync( ptr long long long ) wine_cudaMemsetAsync
@  stdcall cudaMemset2DAsync( ptr long long long long long ) wine_cudaMemset2DAsync
@  stdcall cudaMemset3DAsync( long long long long ) wine_cudaMemset3DAsync
@  stdcall cudaPointerGetAttributes( ptr ptr ) wine_cudaPointerGetAttributes
@  stdcall cudaDeviceCanAccessPeer( ptr long long ) wine_cudaDeviceCanAccessPeer
@  stdcall cudaDeviceEnablePeerAccess( long long ) wine_cudaDeviceEnablePeerAccess
@  stdcall cudaDeviceDisablePeerAccess( long ) wine_cudaDeviceDisablePeerAccess
@  stdcall cudaBindSurfaceToArray( ptr ptr ptr ) wine_cudaBindSurfaceToArray
@  stdcall cudaGetSurfaceReference( ptr ptr ) wine_cudaGetSurfaceReference
@  stdcall cudaGetExportTable( ptr ptr ) wine_cudaGetExportTable
