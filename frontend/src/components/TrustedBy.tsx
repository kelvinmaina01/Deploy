import React from 'react';

const brands = ["Meta", "Microsoft", "Mistral", "Google", "Hugging Face", "Alibaba"];

const TrustedBy: React.FC = () => {
    return (
        <div className="trusted-by-section">
            <p className="trusted-text">Powering the next generation of AI systems</p>
            <div className="ticker-wrapper">
                <div className="ticker-content">
                    {brands.concat(brands).map((brand, i) => (
                        <div key={i} className="brand-logo-alt">{brand}</div>
                    ))}
                </div>
            </div>
        </div>
    );
};

export default TrustedBy;
