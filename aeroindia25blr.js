import React from 'react';

const LinkedInBanner = () => {
  return (
    <div className="relative w-full h-64 bg-gradient-to-r from-blue-900 to-blue-800 overflow-hidden">
      {/* Background elements */}
      <div className="absolute top-0 right-0 w-1/3 h-full opacity-10">
        <svg viewBox="0 0 200 200" className="w-full h-full">
          <path d="M50,0 L150,0 L200,50 L200,150 L150,200 L50,200 L0,150 L0,50 Z" fill="white"/>
        </svg>
      </div>
      
      {/* Main content container */}
      <div className="absolute inset-0 flex flex-col justify-center px-8">
        {/* Logo */}
        <div className="absolute top-4 right-8 bg-white rounded-full p-2 shadow-lg">
          <svg className="w-12 h-12" viewBox="0 0 800 800" fill="none">
            <path d="M400 50 C195 50 50 195 50 400s145 350 350 350 350-145 350-350S605 50 400 50zm0 650c-165 0-300-135-300-300s135-300 300-300 300 135 300 300-135 300-300 300z" fill="#718096"/>
            <path d="M400 150c-138 0-250 112-250 250s112 250 250 250 250-112 250-250-112-250-250-250zm0 450c-110 0-200-90-200-200s90-200 200-200 200 90 200 200-90 200-200 200z" fill="#718096"/>
            <path d="M400 250v300M250 400h300" fill="none" stroke="#718096" strokeWidth="50"/>
          </svg>
        </div>
        
        {/* Event details */}
        <div className="text-blue-200 font-medium mb-2">
          Meet us at Aero India 2025 | February 10-14 | Bangalore
        </div>
        
        {/* Main headline */}
        <div className="text-white text-3xl font-bold mb-4">
          Zero Downtime: End-to-End Smart Factory Solution
        </div>
        
        {/* Subheadline */}
        <div className="text-blue-200 text-lg mb-2">
          Complete IIoT Package: Smart Sensors • Advanced Controllers • AI Analytics
        </div>
        
        {/* Benefits */}
        <div className="text-blue-100 text-sm mb-4">
          40% Lower Maintenance Costs | Real-Time Production Monitoring | Aerospace-Grade Hardware
        </div>
        
        {/* Company info and CTA */}
        <div className="flex items-center justify-between">
          <div className="text-blue-100">
            Gompa Tech | Integrated Hardware-Software Manufacturing Solutions
          </div>
          <div className="bg-blue-600 text-white px-4 py-2 rounded-lg font-medium">
            Let's Connect at Aero India
          </div>
        </div>
      </div>
    </div>
  );
};

export default LinkedInBanner;